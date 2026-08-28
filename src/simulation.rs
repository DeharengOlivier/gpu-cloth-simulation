//! The cloth simulation itself: the particle buffers, the compute pipeline, one
//! physics step.
//!
//! This module knows nothing about windows, egui, or the `App` trait. It takes a
//! `wgpu::Device` and a `wgpu::Queue` and nothing else, which is what makes the
//! physics exercisable without a window: [`ClothSimulation::read_particles`]
//! copies the current state back to the CPU so the invariants the shader is
//! supposed to hold can be asserted rather than looked at.
//!
//! The dependency arrow points inward: `app` depends on `simulation`, never the
//! reverse.

use wgpu_bootstrap::wgpu::{self, util::DeviceExt};

/// GPU workgroup size (threads per workgroup).
///
/// The GPU schedules compute threads in workgroups that run together. 256 is a
/// common, widely supported size.
pub const WORKGROUP_SIZE: u32 = 256;

/// Fixed simulation time step, in seconds.
///
/// A fixed step (rather than the frame's variable delta time) keeps the
/// integration deterministic and stable regardless of frame rate. 0.0016 s is
/// about 1/625, so roughly 625 physics iterations per simulated second.
pub const FIXED_TIME_STEP_SECONDS: f32 = 0.0016;

/// Height, in world units, at which the flat cloth is released.
pub const INITIAL_CLOTH_HEIGHT: f32 = 0.5;

/// How far a spring may stretch past its rest length before the positional
/// constraint pulls its endpoints back.
///
/// The value is bracketed by two measurements taken over the whole range the UI
/// offers (grid 256 and 512, spacing 0.002 to 0.02), with the constraint
/// disabled:
///
/// - a sheet **at rest** stretches up to 1.365x under its own weight (grid 512,
///   spacing 0.002, a shear spring). The cap must sit above that. A cap below it
///   is unsatisfiable at rest, so the constraint fires on every step forever and
///   the correction it feeds back into the velocity accumulates without bound:
///   at 1.1 the sheet reached 5.2e4 units per second and went non-finite.
/// - the **transient**, as the falling sheet snaps taut on the sphere, reaches
///   3.04x. Real cloth does not stretch by 200%, so there is genuine work here.
///
/// 1.5 is the midpoint in spirit: 10% of headroom above the worst resting state,
/// so the constraint never shapes the drape, and well under the transient, so it
/// still clips it. A runaway passes it within a step or two.
pub const MAX_SPRING_STRETCH: f32 = 1.5;

/// Relaxation passes of the positional constraint per physics step.
///
/// One pass moves each endpoint half of its excess, so a single pass leaves a
/// residue behind. The value below is the smallest that held the cap in the
/// harness at the tightest spacing the UI offers; see `tests/physics.rs`.
pub const CONSTRAINT_ITERATIONS: u32 = 2;

/// Coulomb friction coefficient between the cloth and the obstacle sphere.
pub const DEFAULT_FRICTION: f32 = 0.8;

/// One cloth particle, as both the CPU and the GPU see it.
///
/// `#[repr(C)]` gives a stable layout so the two agree on field placement, and
/// bytemuck lets the struct be reinterpreted as raw bytes for upload. Both
/// fields are `vec4` rather than `vec3` because WGSL aligns a `vec3` to 16 bytes
/// anyway: spelling the padding out keeps the two declarations identical.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    /// Particle position (x, y, z) plus one float of padding.
    pub position: [f32; 4],
    /// Particle velocity (vx, vy, vz) plus one float of padding.
    pub speed: [f32; 4],
}

/// Physics parameters, uploaded once as a uniform buffer.
///
/// Uniform means constant for the whole dispatch and readable by every
/// invocation, so the simulation can be tuned without recompiling the shader.
/// IMPORTANT: the field order must match `PhysicsParams` in `compute.wgsl`,
/// because the mapping is by byte offset, not by name.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhysicsParams {
    /// Structural spring stiffness (direct neighbours).
    pub structural_k: f32,
    /// Shear spring stiffness (diagonal neighbours).
    pub shear_k: f32,
    /// Bend spring stiffness (two cells apart).
    pub bend_k: f32,
    /// Damping coefficient, which stops the cloth oscillating forever.
    pub damping: f32,
    /// Mass of one particle.
    pub mass: f32,
    /// Spring rest length: the distance at which the grid is relaxed.
    pub rest_length: f32,
    /// Integration time step, in seconds.
    pub dt: f32,
    /// Coulomb friction coefficient against the obstacle sphere.
    pub friction: f32,
    /// Radius of the obstacle sphere, centred on the origin.
    pub sphere_radius: f32,
    /// How far a spring may stretch past its rest length; see
    /// [`MAX_SPRING_STRETCH`].
    pub max_spring_stretch: f32,
    /// Particles per side of the square grid, which the shader needs to turn a
    /// flat index into a (row, column).
    pub grid_size: u32,
}

/// Everything needed to build a simulation.
#[derive(Clone, Copy, Debug)]
pub struct ClothConfig {
    /// Particles per side of the square grid.
    pub grid_size: u32,
    /// Distance between adjacent particles, which is also the spring rest length.
    pub spacing: f32,
    /// Height at which the cloth is released.
    pub initial_height: f32,
    /// Radius of the obstacle sphere.
    pub sphere_radius: f32,
    /// Relaxation passes of the positional constraint per step.
    ///
    /// Defaults to [`CONSTRAINT_ITERATIONS`]. It is a field rather than a
    /// constant so the harness can run the simulation with the constraint off
    /// and compare, which is how the cap is shown to be doing anything.
    pub constraint_iterations: u32,
    /// How far a spring may stretch before the constraint pulls it back.
    ///
    /// Defaults to [`MAX_SPRING_STRETCH`]. It reaches the shader as a uniform
    /// rather than a substituted literal, so a test can turn the constraint off
    /// and compare against a run with it on.
    pub max_spring_stretch: f32,
    /// Coulomb friction coefficient between the cloth and the obstacle sphere.
    ///
    /// 0 is frictionless, 1 lets friction cancel a tangential force as large as
    /// the normal one.
    pub friction: f32,
}

impl Default for ClothConfig {
    fn default() -> Self {
        Self {
            grid_size: 256,
            spacing: 0.006,
            initial_height: INITIAL_CLOTH_HEIGHT,
            sphere_radius: 0.4,
            constraint_iterations: CONSTRAINT_ITERATIONS,
            max_spring_stretch: MAX_SPRING_STRETCH,
            friction: DEFAULT_FRICTION,
        }
    }
}

impl ClothConfig {
    /// The physics parameters this configuration implies.
    ///
    /// `rest_length` is the grid spacing, so a freshly generated cloth starts
    /// relaxed rather than pre-stretched.
    pub fn physics(&self) -> PhysicsParams {
        PhysicsParams {
            structural_k: 6000.0,
            shear_k: 3000.0,
            bend_k: 450.0,
            damping: 0.1,
            mass: 0.1,
            rest_length: self.spacing,
            dt: FIXED_TIME_STEP_SECONDS,
            friction: self.friction,
            sphere_radius: self.sphere_radius,
            max_spring_stretch: self.max_spring_stretch,
            grid_size: resolve_grid_size(self.grid_size),
        }
    }
}

/// The smallest grid that has a spring in it at all.
pub const MIN_GRID_SIZE: u32 = 2;

/// The grid side actually built for a requested one.
///
/// It used to round down to a multiple of [`WORKGROUP_SIZE`] so the dispatch
/// divided evenly, which turned every request under 256 into 256: the slider
/// offered a 64 x 64 sheet and ran a 256 x 256 one. The dispatch now rounds the
/// *workgroup count* up instead and the shader discards the surplus invocations,
/// so any side works and the only adjustment left is the floor below which there
/// are no springs to simulate.
///
/// Complexity: O(1).
pub fn resolve_grid_size(grid_size: u32) -> u32 {
    grid_size.max(MIN_GRID_SIZE)
}

/// Builds the particles of a `rows` x `cols` grid centred on the origin, flat at
/// height `displacement`, every one of them at rest.
///
/// Row-major: `index = row * cols + col`. Complexity: O(rows x cols).
pub fn generate_instances(rows: u32, cols: u32, spacing: f32, displacement: f32) -> Vec<Instance> {
    (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| Instance {
                position: [
                    (col as f32 - cols as f32 / 2.0) * spacing,
                    displacement,
                    (row as f32 - rows as f32 / 2.0) * spacing,
                    0.0, // Padding (vec4 alignment)
                ],
                speed: [0.0, 0.0, 0.0, 0.0], // Start at rest
            })
        })
        .collect()
}

/// The GPU-side cloth: two particle buffers, the compute pipeline, and the
/// ping-pong bookkeeping that keeps a parallel step free of read/write conflicts.
///
/// ```text
/// Step N:   buffer[0] (read) -> compute -> buffer[1] (write)
/// Step N+1: buffer[1] (read) -> compute -> buffer[0] (write)
/// ```
pub struct ClothSimulation {
    buffers: [wgpu::Buffer; 2],
    bind_groups: [wgpu::BindGroup; 2],
    compute_pipeline: wgpu::ComputePipeline,
    constraint_pipeline: wgpu::ComputePipeline,
    grid_size: u32,
    particle_count: u32,
    workgroups: u32,
    constraint_iterations: u32,
}

impl ClothSimulation {
    /// Builds every GPU resource the physics step needs.
    ///
    /// The grid side passes through [`resolve_grid_size`]; [`Self::grid_size`]
    /// reports what was actually built.
    pub fn new(device: &wgpu::Device, config: &ClothConfig) -> Self {
        let grid_size = resolve_grid_size(config.grid_size);
        let instances =
            generate_instances(grid_size, grid_size, config.spacing, config.initial_height);
        let particle_count = instances.len() as u32;

        // Both buffers start from the same state, and both are usable as
        // instanced vertex data so the renderer can draw straight from them.
        let usage =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC;
        let buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer Ping"),
                contents: bytemuck::cast_slice(instances.as_slice()),
                usage,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer Pong"),
                contents: bytemuck::cast_slice(instances.as_slice()),
                usage,
            }),
        ];

        let physics_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Physics Params Buffer"),
            contents: bytemuck::cast_slice(&[config.physics()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                storage_entry(0),
                storage_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Two bind groups over the same two buffers with the read and write roles
        // swapped. Swapping which one is bound is the whole ping-pong mechanism.
        let bind_groups = [
            bind_group(
                device,
                &layout,
                &buffers[0],
                &buffers[1],
                &physics_buffer,
                0,
            ),
            bind_group(
                device,
                &layout,
                &buffers[1],
                &buffers[0],
                &physics_buffer,
                1,
            ),
        ];

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            // The literal WORKGROUP_SIZE in the WGSL source is substituted with
            // the Rust constant, so the two can never disagree.
            source: wgpu::ShaderSource::Wgsl(compute_shader_source().into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = pipeline(device, &pipeline_layout, &shader, "computeMain");
        let constraint_pipeline = pipeline(device, &pipeline_layout, &shader, "constraintMain");

        Self {
            buffers,
            bind_groups,
            compute_pipeline,
            constraint_pipeline,
            grid_size,
            particle_count,
            workgroups: particle_count.div_ceil(WORKGROUP_SIZE),
            constraint_iterations: config.constraint_iterations,
        }
    }

    /// Advances the simulation by one fixed time step.
    ///
    /// Two kinds of dispatch, in this order and deliberately not merged:
    ///
    /// 1. integration, which turns spring forces, gravity and collisions into a
    ///    new position and velocity;
    /// 2. the positional constraint, which pulls back any spring that came out
    ///    of step 1 stretched past [`MAX_SPRING_STRETCH`].
    ///
    /// They cannot share a pass. An invocation may only write its own particle,
    /// so a merged pass compares a post-integration position against
    /// pre-integration neighbours and corrects towards a configuration that
    /// never existed. Splitting them is what makes the cap hold.
    ///
    /// Complexity: O(n) per dispatch with n particles, each invocation touching
    /// a fixed number of neighbours, and 1 + [`CONSTRAINT_ITERATIONS`]
    /// dispatches per step.
    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.dispatch(device, queue, Pass::Integrate);
        for _ in 0..self.constraint_iterations {
            self.dispatch(device, queue, Pass::Constrain);
        }
    }

    /// Runs one pass and swaps the ping-pong pair, so what was just written
    /// becomes what the next pass reads.
    fn dispatch(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, which: Pass) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(which.label()),
                timestamp_writes: None,
            });
            pass.set_pipeline(match which {
                Pass::Integrate => &self.compute_pipeline,
                Pass::Constrain => &self.constraint_pipeline,
            });
            pass.set_bind_group(0, &self.bind_groups[0], &[]);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        self.buffers.swap(0, 1);
        self.bind_groups.swap(0, 1);
    }

    /// The buffer holding the most recently computed state, for rendering.
    pub fn current_buffer(&self) -> &wgpu::Buffer {
        &self.buffers[0]
    }

    /// Particles per side of the grid that was actually built.
    pub fn grid_size(&self) -> u32 {
        self.grid_size
    }

    /// Total number of simulated particles.
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }

    /// Number of workgroups one step dispatches.
    pub fn workgroups(&self) -> u32 {
        self.workgroups
    }

    /// Copies the current particle state back to the CPU.
    ///
    /// This stalls the GPU, so it belongs in tests and benchmarks and nowhere
    /// near the frame loop. It exists so the shader's invariants can be
    /// asserted instead of eyeballed.
    pub fn read_particles(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Instance> {
        let size = (self.particle_count as usize * std::mem::size_of::<Instance>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(self.current_buffer(), 0, &staging, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // The receiver is alive until the poll below returns, so a failure
            // to send would itself be a bug worth surfacing.
            sender.send(result).expect("readback receiver dropped");
        });
        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .expect("readback never completed")
            .expect("readback failed");

        let particles = bytemuck::cast_slice::<u8, Instance>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        particles
    }
}

/// The compute shader source with [`WORKGROUP_SIZE`] substituted in.
///
/// The dispatch size and the shader's `@workgroup_size` must agree, and one is
/// Rust while the other is WGSL, so the number is written once here and injected
/// rather than repeated. Everything else the two sides share travels as a
/// uniform in [`PhysicsParams`], which needs no rewriting of the source.
fn compute_shader_source() -> String {
    include_str!("compute.wgsl").replace("WORKGROUP_SIZE", &WORKGROUP_SIZE.to_string())
}

/// Which of the two passes a dispatch runs.
#[derive(Clone, Copy)]
enum Pass {
    Integrate,
    Constrain,
}

impl Pass {
    fn label(self) -> &'static str {
        match self {
            Self::Integrate => "Integration Pass",
            Self::Constrain => "Constraint Pass",
        }
    }
}

/// Builds one compute pipeline over the shared layout and shader module.
fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(layout),
        module: shader,
        entry_point,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// A read-write storage buffer binding for the compute stage.
fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// One side of the ping-pong pair: `read` at binding 0, `write` at binding 1.
fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    read: &wgpu::Buffer,
    write: &wgpu::Buffer,
    physics: &wgpu::Buffer,
    index: u32,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(if index == 0 {
            "Bind Group Ping"
        } else {
            "Bind Group Pong"
        }),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: read.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: write.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: physics.as_entire_binding(),
            },
        ],
    })
}

// ============================================================================
// UNIT TESTS (CPU-ONLY)
// ============================================================================
//
// These exercise the pure-CPU logic and the CPU/GPU layout invariants. They
// touch no GPU. The physics itself is covered by the headless harness in
// tests/physics.rs, which does.

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    // The CPU uploads these structs as raw bytes and the WGSL side reads them
    // back by byte offset, not by field name. A size or offset mismatch is a
    // silent bug: the shader reads the wrong data and nothing says so.

    #[test]
    fn instance_layout_matches_the_shader() {
        // compute.wgsl declares Instance as two vec4<f32>: 16-byte aligned,
        // 16 bytes each, so 32 bytes with speed at offset 16.
        assert_eq!(size_of::<Instance>(), 32, "Instance is 2 x vec4 = 32 bytes");
        assert_eq!(align_of::<Instance>(), 4);
        assert_eq!(offset_of!(Instance, position), 0);
        assert_eq!(
            offset_of!(Instance, speed),
            16,
            "speed starts at 16 (vec4 alignment)"
        );
    }

    #[test]
    fn physics_params_layout_matches_the_shader() {
        // compute.wgsl declares ten f32 scalars and one u32, all 4 bytes and
        // all consecutive. Field ORDER is load-bearing, so every offset is
        // pinned to catch a reordering.
        assert_eq!(size_of::<PhysicsParams>(), 44, "10 f32 + 1 u32 = 44 bytes");
        assert_eq!(align_of::<PhysicsParams>(), 4);
        assert_eq!(offset_of!(PhysicsParams, structural_k), 0);
        assert_eq!(offset_of!(PhysicsParams, shear_k), 4);
        assert_eq!(offset_of!(PhysicsParams, bend_k), 8);
        assert_eq!(offset_of!(PhysicsParams, damping), 12);
        assert_eq!(offset_of!(PhysicsParams, mass), 16);
        assert_eq!(offset_of!(PhysicsParams, rest_length), 20);
        assert_eq!(offset_of!(PhysicsParams, dt), 24);
        assert_eq!(offset_of!(PhysicsParams, friction), 28);
        assert_eq!(offset_of!(PhysicsParams, sphere_radius), 32);
        assert_eq!(offset_of!(PhysicsParams, max_spring_stretch), 36);
        assert_eq!(offset_of!(PhysicsParams, grid_size), 40);
    }

    #[test]
    fn the_shader_declares_the_fields_in_the_order_the_struct_does() {
        // Reading the WGSL source keeps this honest even if nobody reruns the
        // simulation after reordering the Rust struct.
        let shader = include_str!("compute.wgsl");
        let body = shader
            .split("struct PhysicsParams {")
            .nth(1)
            .expect("compute.wgsl must declare PhysicsParams");
        let declared: Vec<&str> = body
            .split('}')
            .next()
            .expect("unterminated PhysicsParams declaration")
            .lines()
            .filter_map(|line| line.trim().split(':').next())
            .filter(|name| !name.is_empty())
            .collect();
        assert_eq!(
            declared,
            [
                "structural_k",
                "shear_k",
                "bend_k",
                "damping",
                "mass",
                "rest_length",
                "dt",
                "friction",
                "sphere_radius",
                "max_spring_stretch",
                "grid_size",
            ]
        );
    }

    #[test]
    fn the_shader_binds_exactly_what_the_pipeline_declares() {
        // Three bindings: the two particle buffers and the physics uniform. A
        // binding declared on one side only is a pipeline creation failure at
        // startup, which is a poor place to find out.
        let shader = include_str!("compute.wgsl");
        for binding in ["@binding(0)", "@binding(1)", "@binding(2)"] {
            assert!(shader.contains(binding), "compute.wgsl must use {binding}");
        }
        assert!(
            !shader.contains("@binding(3)"),
            "the layout declares three bindings, so the shader must not use a fourth"
        );
    }

    // ---- the grid side the caller asks for ----

    #[test]
    fn the_requested_grid_side_is_the_one_used() {
        // It used to round down to a multiple of WORKGROUP_SIZE, so every side
        // under 256 silently became 256.
        for requested in [2u32, 3, 63, 64, 100, 255, 256, 300, 512, 1000] {
            assert_eq!(resolve_grid_size(requested), requested);
        }
    }

    #[test]
    fn a_grid_too_small_to_hold_a_spring_is_raised_to_one_that_can() {
        for requested in [0, 1] {
            assert_eq!(resolve_grid_size(requested), MIN_GRID_SIZE);
        }
    }

    #[test]
    fn the_dispatch_covers_every_particle_and_wastes_at_most_one_workgroup() {
        // Rounding up is what lets any grid side run, and the shader discards
        // the surplus invocations. Rounding down would leave the last particles
        // unstepped, which is the failure this pins.
        for side in [2u32, 3, 63, 64, 100, 255, 256, 300, 512] {
            let particles = side * side;
            let workgroups = particles.div_ceil(WORKGROUP_SIZE);
            assert!(
                workgroups * WORKGROUP_SIZE >= particles,
                "side={side} leaves {} particles unstepped",
                particles - workgroups * WORKGROUP_SIZE
            );
            assert!(
                (workgroups - 1) * WORKGROUP_SIZE < particles,
                "side={side} dispatches a workgroup with nothing in it"
            );
        }
    }

    // ---- Grid generation ----

    #[test]
    fn generate_instances_has_the_expected_count() {
        assert_eq!(generate_instances(4, 3, 0.01, 0.5).len(), 12);
    }

    #[test]
    fn generate_instances_is_row_major() {
        let (rows, cols, spacing, height) = (3u32, 5u32, 0.01f32, 0.5f32);
        let instances = generate_instances(rows, cols, spacing, height);
        for row in 0..rows {
            for col in 0..cols {
                let particle = instances[(row * cols + col) as usize];
                let expected_x = (col as f32 - cols as f32 / 2.0) * spacing;
                let expected_z = (row as f32 - rows as f32 / 2.0) * spacing;
                assert!((particle.position[0] - expected_x).abs() < 1e-9);
                assert_eq!(particle.position[1], height);
                assert!((particle.position[2] - expected_z).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn generate_instances_start_at_rest_with_zero_padding() {
        for particle in generate_instances(2, 2, 0.01, 0.5) {
            assert_eq!(particle.position[3], 0.0, "position padding must be zero");
            assert_eq!(particle.speed, [0.0; 4], "every particle starts at rest");
        }
    }

    #[test]
    fn generate_instances_centres_the_grid_on_the_origin_axes() {
        // The formula is (col - cols/2) * spacing, so col == cols/2 sits on
        // X = 0. For an even side that is half a cell off true centre; this
        // pins the actual behaviour rather than an idealised one.
        let (rows, cols, spacing) = (4u32, 4u32, 0.01f32);
        let instances = generate_instances(rows, cols, spacing, 0.5);
        let centre = ((rows / 2) * cols + (cols / 2)) as usize;
        assert_eq!(instances[centre].position[0], 0.0);
        assert_eq!(instances[centre].position[2], 0.0);

        let span = instances[(cols - 1) as usize].position[0] - instances[0].position[0];
        assert!((span - (cols as f32 - 1.0) * spacing).abs() < 1e-6);
    }

    #[test]
    fn the_default_configuration_is_dispatch_safe() {
        let config = ClothConfig::default();
        let side = resolve_grid_size(config.grid_size);
        assert_eq!(side, 256);
        assert!(side * side >= WORKGROUP_SIZE);
    }

    #[test]
    fn the_default_cloth_starts_relaxed() {
        // rest_length must equal the spacing, or the cloth begins pre-stretched
        // and snaps on the first step.
        let config = ClothConfig::default();
        assert_eq!(config.physics().rest_length, config.spacing);
    }
}
