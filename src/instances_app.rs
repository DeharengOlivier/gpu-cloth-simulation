// ============================================================================
// IMPORTS AND DEPENDENCIES
// ============================================================================

// wgpu_bootstrap: thin framework over WebGPU/wgpu (window, device, frame loop, egui).
use wgpu_bootstrap::{
    cgmath::{self, InnerSpace}, // 3D math (vectors, matrices)
    egui,                       // Immediate-mode GUI library
    util::{
        geometry::icosphere,    // Generates a subdivided sphere (icosphere)
        orbit_camera::{CameraUniform, OrbitCamera}, // Orbit camera for 3D navigation
    },
    wgpu::{self, util::DeviceExt}, // WebGPU API
    App, Context,                  // Framework trait and context type
};
use std::time::{Duration, Instant}; // Timing for the physics loop

// ============================================================================
// GPU DATA STRUCTURES
// ============================================================================

/// One mesh vertex.
///
/// #[repr(C)] gives a stable, C-compatible memory layout so the CPU and GPU
/// agree on field placement. bytemuck::Pod / Zeroable allow reinterpreting the
/// struct as raw bytes for upload to a GPU buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],  // Position (x, y, z) in 3D space
    normal: [f32; 3],    // Normal for lighting (unused by the cloth, used by the sphere)
    color: [f32; 3],     // RGB color (each component in 0.0..=1.0)
}

impl Vertex {
    /// Describes to the GPU how to read Vertex data out of the vertex buffer:
    /// the stride between vertices, each attribute's byte offset, and how to step
    /// through the buffer.
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            // Bytes between consecutive vertices: 9 floats x 4 bytes = 36 bytes.
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,

            // Vertex step mode: advance once per vertex (Instance mode is used by Instance).
            step_mode: wgpu::VertexStepMode::Vertex,

            attributes: &[
                // Attribute 0: position
                wgpu::VertexAttribute {
                    offset: 0,  // Start of the struct
                    shader_location: 0,  // Matches @location(0) in the shader
                    format: wgpu::VertexFormat::Float32x3,  // 3 floats (x, y, z)
                },
                // Attribute 1: normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,  // After position (12 bytes)
                    shader_location: 1,  // @location(1)
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Attribute 2: color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,  // After position + normal (24 bytes)
                    shader_location: 2,  // @location(2)
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// One cloth particle.
///
/// Instancing: instead of duplicating the small visualization sphere for every
/// particle, the shared mesh is drawn once per Instance, translated to that
/// particle's position. Each Instance therefore only carries per-particle data.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    position: [f32; 4],  // Particle position (x, y, z) + 1 float of padding
    speed: [f32; 4],     // Particle velocity (vx, vy, vz) + padding
                         // vec4 is used so each field is 16-byte aligned for the GPU,
                         // matching the vec4 layout of Instance in compute.wgsl.
}

impl Instance {
    /// Vertex buffer layout for instanced rendering.
    ///
    /// Unlike Vertex, step_mode is Instance: this data advances once per drawn
    /// instance rather than once per vertex.
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,

            // Same data for every vertex of an instance, different per instance.
            step_mode: wgpu::VertexStepMode::Instance,

            attributes: &[
                // Instance position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,  // @location(3) in the shader
                    format: wgpu::VertexFormat::Float32x3,  // The 4th (padding) component is ignored
                },
                // Instance velocity (not used by the render shader, but kept for the compute layout)
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32;3]>() as wgpu::BufferAddress,
                    shader_location: 4,  // @location(4) in the shader
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Time uniform. Bound to the compute pass for layout compatibility but not read
/// by the current simulation step.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TimeUniform {
    generation_duration: f32,  // Duration of a generated frame
}

/// Physics simulation parameters, uploaded as a uniform buffer.
///
/// Uniform = constant for the whole compute dispatch and readable by every
/// invocation, so the simulation can be tuned without recompiling the shaders.
/// IMPORTANT: the field order here must match PhysicsParams in compute.wgsl,
/// since the CPU/GPU mapping is by byte offset (bytemuck), not by name.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PhysicsParams {
    structural_k: f32,    // Structural spring stiffness (direct neighbours)
    shear_k: f32,         // Shear spring stiffness (diagonals)
    bend_k: f32,          // Bend spring stiffness (two cells apart)
    damping: f32,         // Damping coefficient (prevents endless oscillation)
    mass: f32,            // Mass of each particle (for F = m*a)
    rest_length: f32,     // Spring rest length (natural distance between particles)
    dt: f32,              // Time step of the simulation
    friction: f32,        // Friction coefficient with the sphere
    sphere_radius: f32,   // Radius of the collision sphere
}

// ============================================================================
// APPLICATION SETTINGS
// ============================================================================

/// User-tunable configuration exposed through the egui panel.
///
/// Some changes (grid_size, spacing, point_size) require rebuilding the GPU
/// buffers; colors can be updated in place.
#[derive(Clone)]
pub struct ClothSettings {
    pub grid_size: u32,        // N for the NxN grid (particles per side)
    pub spacing: f32,          // Distance between adjacent particles
    pub point_size: f32,       // Render size of each particle (radius of its mini-sphere)
    pub cloth_color: [f32; 3], // Cloth RGB color
    pub sphere_color: [f32; 3],// Central sphere RGB color
}

impl Default for ClothSettings {
    /// Default parameter values.
    fn default() -> Self {
        Self {
            grid_size: 256,              // 256x256 grid = 65,536 particles
            spacing: 0.006,              // 6 mm between particles
            point_size: 0.0033,          // Visualization sphere radius
            cloth_color: [1.0, 0.0, 0.0],// Red
            sphere_color: [0.5, 0.5, 0.5],// Grey
        }
    }
}

// ============================================================================
// MAIN APPLICATION STRUCTURE
// ============================================================================

/// Core of the cloth simulation. Holds every GPU resource and bit of state
/// needed to simulate and render the cloth. Implements wgpu_bootstrap's App trait.
pub struct InstanceApp {
    // === Cloth GPU buffers ===
    vertex_buffer: wgpu::Buffer,        // Geometry of the mini-spheres (vertices)
    instance_buffer: [wgpu::Buffer; 2], // Particle position/velocity (ping-pong pair)
    index_buffer: wgpu::Buffer,         // Triangle indices

    // === Render and compute pipelines ===
    render_pipeline: wgpu::RenderPipeline,   // Draws the cloth
    compute_pipeline: wgpu::ComputePipeline, // Runs the physics step

    // === Metadata ===
    num_indices: u32,      // Number of indices to draw per cloth particle
    num_instances: u32,    // Number of particles (instances)

    // === Camera ===
    camera: OrbitCamera,   // Mouse-controllable orbit camera
    last_generation: Instant, // Timestamp of the last simulation step (timing)

    // === Bind groups ===
    // Bind groups connect buffers to the shaders. Two are kept for the ping-pong
    // scheme: index 0 is the one used this step, swapped after each step.
    bind_group: [wgpu::BindGroup; 2],

    // === Central sphere GPU buffers ===
    sphere_index_buffer: wgpu::Buffer,
    sphere_vertex_buffer: wgpu::Buffer,
    num_sphere_indices: u32,
    sphere_render_pipeline: wgpu::RenderPipeline,

    // === UI state ===
    settings: ClothSettings,         // Settings currently applied
    pending_settings: ClothSettings, // Settings edited in the UI, not yet applied
    needs_rebuild: bool,             // Set when a rebuild is required
    paused: bool,                    // Pause/play state of the simulation
}

// ============================================================================
// CLOTH GRID GENERATION
// ============================================================================

/// Builds the grid of particles that make up the cloth.
///
/// Separation of concerns:
/// - Vertices: the base mesh (a small sphere) shared by ALL particles.
/// - Instances: the unique position of each particle in the grid.
///
/// The GPU draws the same mesh (vertices) once per instance, at each instance's
/// position. This is instanced rendering.
///
/// # Arguments
/// * `rows` - Number of particle rows
/// * `cols` - Number of particle columns
/// * `spacing` - Distance between adjacent particles
/// * `displacement` - Initial height of the cloth
/// * `sphere_scale` - Radius of the visualization mini-spheres
/// * `cloth_color` - RGB color of the particles
///
/// # Returns
/// (vertices, index_buffer, instances, instances_copy, indices)
fn generate_grid(
    context: &Context,
    rows: u32,
    cols: u32,
    spacing: f32,
    displacement: f32,
    sphere_scale: f32,
    cloth_color: [f32; 3],
) -> (Vec<Vertex>, wgpu::Buffer, Vec<Instance>, Vec<Instance>, Vec<u32>) {
    // Generate a subdivided sphere (icosphere, subdivision level 2).
    // Higher level = smoother sphere but more triangles per particle.
    let (positions, indices) = icosphere(2);

    // Build the vertices: scale the unit-sphere positions and tag them with a
    // (zero) normal and the cloth color.
    let vertices: Vec<Vertex> = positions
        .iter()
        .map(|position| Vertex {
            position: (*position * sphere_scale).into(), // Scale the sphere
            normal: [0.0, 0.0, 0.0],                     // Normal unused for the cloth
            color: cloth_color,                          // Cloth color
        })
        .collect();

    // Index buffer: triangle indices referencing the vertices above, so vertices
    // are not duplicated (e.g. triangle (0,1,2) reuses vertices 0, 1 and 2).
    let index_buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            // bytemuck::cast_slice reinterprets Vec<u32> as raw &[u8] for upload.
            contents: bytemuck::cast_slice(indices.as_slice()),
            // INDEX usage: this buffer feeds indexed draws.
            usage: wgpu::BufferUsages::INDEX,
        });

    // Build the particle grid: one Instance (position + velocity) per particle,
    // centered on the origin. (Pure-CPU logic, see generate_instances.)
    let instances: Vec<Instance> = generate_instances(rows, cols, spacing, displacement);

    // Identical copy used to initialize the second ping-pong buffer (see below).
    let instances_copy = instances.clone();

    (vertices, index_buffer, instances, instances_copy, indices)
}

// ============================================================================
// SIMULATION CONSTANTS
// ============================================================================

/// Fixed simulation time step, in seconds.
///
/// Using a fixed time step (rather than the frame's variable delta_time) keeps
/// the integration deterministic and stable regardless of frame rate.
/// 0.0016 s is about 1/625, i.e. ~625 physics iterations per simulated second.
/// (Name kept as-is to avoid a behavior-neutral rename of a referenced symbol.)
const TAYME: f32 = 0.0016;

/// GPU workgroup size (threads per workgroup).
///
/// The GPU schedules compute threads in workgroups that run together. 256 is a
/// common, widely supported size. grid_size is rounded down to a multiple of
/// this value so every particle is covered by a whole number of workgroups.
const WORKGROUP_SIZE: u32 = 256;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Rounds a requested grid side length down to a multiple of WORKGROUP_SIZE and
/// clamps it to at least one full workgroup.
///
/// The compute dispatch covers `grid_size / WORKGROUP_SIZE` workgroups of
/// WORKGROUP_SIZE threads each, so the side length must be a whole multiple of
/// WORKGROUP_SIZE for every particle to be processed (and never zero). This is
/// pure integer arithmetic, extracted so it can be unit-tested without a GPU.
fn round_grid_size(grid_size: u32) -> u32 {
    let rounded = (grid_size / WORKGROUP_SIZE) * WORKGROUP_SIZE;
    rounded.max(WORKGROUP_SIZE) // Minimum one workgroup
}

/// Builds the per-particle instances (position + velocity) for an `rows` x `cols`
/// grid centered on the origin, at height `displacement`, with the given spacing.
///
/// This is the pure-CPU part of `generate_grid` (no GPU device needed), separated
/// so the grid layout can be unit-tested. Behavior is identical to the original
/// inline loop: row-major order (index = row * cols + col), centered on X/Z, the
/// 4th position component is padding, and every particle starts at rest.
fn generate_instances(rows: u32, cols: u32, spacing: f32, displacement: f32) -> Vec<Instance> {
    (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| Instance {
                position: [
                    // X: centered, spaced by 'spacing'
                    (col as f32 - cols as f32 / 2.0) * spacing,
                    // Y: initial height
                    displacement,
                    // Z: centered, spaced by 'spacing'
                    (row as f32 - rows as f32 / 2.0) * spacing,
                    0.0, // Padding (vec4 alignment)
                ],
                speed: [0.0, 0.0, 0.0, 0.0], // Start at rest
            })
        })
        .collect()
}

/// Builds the vertices for the central obstacle sphere.
///
/// This sphere is static and acts as an obstacle for the cloth. Unlike the
/// cloth it does NOT use instancing: it is drawn exactly once.
fn create_sphere_vertices(sphere_radius: f32, sphere_color: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
    // Subdivision level 3: more detailed than the cloth's mini-spheres.
    let (positions, indices) = icosphere(3);
    let vertices: Vec<Vertex> = positions
        .iter()
        .map(|position| {
            // On a unit sphere the outward normal equals the normalized position.
            let normal = position.normalize();
            Vertex {
                position: (normal * sphere_radius).into(), // Place on the surface
                normal: normal.into(),                     // Normal for lighting
                color: sphere_color,                       // Sphere color
            }
        })
        .collect();
    (vertices, indices)
}

// ============================================================================
// APPLICATION IMPLEMENTATION
// ============================================================================

impl InstanceApp {
    /// Main constructor: initializes with default settings.
    pub fn new(context: &Context) -> Self {
        let settings = ClothSettings::default();
        Self::create_with_settings(context, settings)
    }

    /// Constructor with explicit settings. Does all the initialization work:
    /// 1. Create GPU buffers
    /// 2. Compile the shaders
    /// 3. Configure the pipelines
    /// 4. Bind the resources (bind groups)
    fn create_with_settings(context: &Context, settings: ClothSettings) -> Self {
        // === STEP 1: VALIDATE AND GENERATE THE GRID ===

        // Round grid_size down to a multiple of WORKGROUP_SIZE so the dispatch
        // covers exactly all particles, then clamp to at least one workgroup.
        let grid_size = round_grid_size(settings.grid_size);

        // Generate the vertices (mini-sphere geometry) and instances (particles).
        let (vertices, index_buffer, instances, instances_copy, indices) = generate_grid(
            &context,
            grid_size,       // Rows
            grid_size,       // Columns
            settings.spacing,// Distance between particles
            0.5,             // Initial height
            settings.point_size, // Visualization sphere radius
            settings.cloth_color, // Color
        );

        let num_indices = indices.len() as u32;     // Indices to draw per particle
        let num_instances = instances.len() as u32; // Total particle count

        // === STEP 2: UNIFORM BUFFERS ===

        // TimeUniform: currently unused, kept for bind-group compatibility.
        let time_uniform = TimeUniform {
            generation_duration: Duration::new(0, 1_000_000).as_secs_f32(),
        };

        // Uniform buffer (constant during a draw/compute call).
        // COPY_DST lets us update it later via queue.write_buffer().
        let time_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Time Uniform Buffer"),
            contents: bytemuck::cast_slice(&[time_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === STEP 3: VERTEX AND INSTANCE BUFFERS ===

        // Vertex buffer: mini-sphere geometry shared by every particle.
        // VERTEX = used as vertex input; COPY_DST = color can be updated in place.
        let vertex_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        // === PING-PONG BUFFERS ===
        //
        // The compute shader must read the current particle state and write the
        // next one. Two buffers alternate read/write roles to avoid reading and
        // writing the same memory in a single parallel pass:
        //
        //   Step N:   Buffer[0] (read) -> Compute -> Buffer[1] (write)
        //   Step N+1: Buffer[1] (read) -> Compute -> Buffer[0] (write)
        //
        // Both buffers also serve as instance vertex buffers for rendering.
        let instance_buffer = [
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer Ping"),
                    contents: bytemuck::cast_slice(&instances.as_slice()),
                    // STORAGE = read/write in the compute shader.
                    // VERTEX = usable as instanced vertex data.
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
                }),
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer Pong"),
                    contents: bytemuck::cast_slice(&instances_copy.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
                }),
        ];

        // === STEP 4: PHYSICS PARAMETERS ===

        let (_positions, _indices) = icosphere(3); // Generated but unused here
        let sphere_radius = 0.4; // Obstacle sphere radius

        // Force coefficients and behavior of the simulation.
        let physics_params = PhysicsParams {
            structural_k: 4000.0 * 1.5,  // Structural stiffness (direct links)
            shear_k: 2000.0 * 1.5,       // Shear stiffness (diagonals)
            bend_k: 300.0 * 1.5,         // Bend stiffness (two cells apart)
            damping: 0.1,                // Damping (dissipates energy)
            mass: 0.1,                   // Mass per particle
            rest_length: settings.spacing, // Must equal spacing so the grid starts relaxed
            dt: TAYME,                   // Time step
            friction: 0.8,               // Friction with the sphere
            sphere_radius: sphere_radius,// Collision radius
        };

        // Uniform buffer for the physics parameters.
        let physics_buffer = context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Physics Params Buffer"),
                contents: bytemuck::cast_slice(&[physics_params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        // === STEP 5: OBSTACLE SPHERE ===

        // Build the sphere with the color from the settings.
        let (sphere_vertices, sphere_indices) = create_sphere_vertices(sphere_radius, settings.sphere_color);

        // Sphere buffers (static, no instancing).
        let sphere_vertex_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sphere Vertex Buffer"),
                contents: bytemuck::cast_slice(sphere_vertices.as_slice()),
                // COPY_DST: lets the color be updated in place.
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        let sphere_index_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sphere Index Buffer"),
                contents: bytemuck::cast_slice(sphere_indices.as_slice()),
                usage: wgpu::BufferUsages::INDEX,
            });

        // === STEP 6: SHADER COMPILATION ===

        // Render shader (vertex + fragment).
        let shader = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });

        // Compute shader: the GPU physics step. The literal "WORKGROUP_SIZE" in
        // the WGSL source is substituted with the actual value before compilation.
        let compute_shader = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("compute.wgsl")
                        .replace("WORKGROUP_SIZE", &format!("{}", WORKGROUP_SIZE))
                        .into()
                ),
            });

        // === STEP 7: BIND GROUP LAYOUTS ===
        //
        // A bind group layout declares which resources a shader expects and how
        // they are accessed; the bind group itself supplies the concrete buffers.

        // Layout for the camera (view + projection matrices).
        let camera_bind_group_layout = context
            .device()
            .create_bind_group_layout(&CameraUniform::desc());

        // Layout for the compute shader, with 4 bindings:
        let instance_bind_group_layout = context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // Binding 0: instance read buffer (ping)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE, // Compute stage only
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // read_write storage
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: instance write buffer (pong)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: time uniform (currently unused)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform, // Uniform = read-only, constant
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: physics parameters
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        // === STEP 8: PIPELINE LAYOUTS ===
        //
        // A pipeline layout fixes which bind groups a pipeline uses and in what order.

        // Render pipeline layout (cloth): uses only the camera at bind group 0.
        let pipeline_layout = context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout], // Bind group 0 = camera
                push_constant_ranges: &[], // No push constants
            });

        // Compute pipeline layout (physics): instance buffers + params at bind group 0.
        let compute_pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&instance_bind_group_layout], // Bind group 0 = instances + params
            push_constant_ranges: &[], // No push constants
        });

        // === STEP 9: RENDER PIPELINE ===
        //
        // The render pipeline is the full configuration that turns vertices into
        // pixels on screen.
        let render_pipeline = context
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout), // Layout defined above

                // Vertex stage: transforms 3D positions into clip space.
                vertex: wgpu::VertexState {
                    module: &shader,              // Compiled WGSL module
                    entry_point: "vs_main",       // Vertex entry point
                    buffers: &[Vertex::desc(), Instance::desc()], // Two buffers: geometry + instances
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },

                // Fragment stage: computes each pixel's color.
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",       // Fragment entry point
                    targets: &[Some(wgpu::ColorTargetState {
                        format: context.format(),          // Surface format (e.g. BGRA8)
                        blend: Some(wgpu::BlendState::REPLACE), // No blending, overwrite
                        write_mask: wgpu::ColorWrites::ALL,     // Write full RGBA
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),

                // Primitive assembly (triangles).
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList, // Triangle list
                    strip_index_format: None,                        // Not a triangle strip
                    front_face: wgpu::FrontFace::Ccw,                // Counter-clockwise = front face
                    cull_mode: Some(wgpu::Face::Back),               // Cull back faces (optimization)
                    polygon_mode: wgpu::PolygonMode::Fill,           // Filled triangles (not wireframe)
                    unclipped_depth: false,
                    conservative: false,
                },

                // Depth test: discards fragments hidden behind closer ones.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: context.depth_stencil_format(),
                    depth_write_enabled: true,                      // Write to the depth buffer
                    depth_compare: wgpu::CompareFunction::Less,     // Keep the nearest fragment
                    stencil: wgpu::StencilState::default(),         // No stencil
                    bias: wgpu::DepthBiasState::default(),
                }),

                // Multisampling (anti-aliasing) disabled here.
                multisample: wgpu::MultisampleState {
                    count: 1,                       // No MSAA
                    mask: !0,                       // All samples active
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,  // No stereo rendering (VR)
                cache: None,      // No pipeline cache
            });

        // === STEP 10: CAMERA SETUP ===
        //
        // An orbit camera rotates around a central point, letting the user view
        // the scene from any angle.
        let aspect = context.size().x / context.size().y; // Window width/height ratio
        let mut camera = OrbitCamera::new(
            context,
            45.0,    // Field of view, in degrees
            aspect,  // Aspect ratio (avoids distortion)
            0.1,     // Near plane
            100.0    // Far plane
        );
        // Place the camera 1.5 units from the center (polar coordinates).
        camera
            .set_polar(cgmath::point3(1.5, 0.0, 0.0))
            .update(context); // Recompute the view/projection matrices

        // === STEP 11: COMPUTE PIPELINE ===
        //
        // A compute pipeline configures a parallel GPU computation. It is simpler
        // than a render pipeline: no vertex/fragment stages, just the kernel.
        let compute_pipeline = context
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout), // Layout with instances + params
                module: &compute_shader,                // Physics WGSL module
                entry_point: "computeMain",             // @compute entry point
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // === STEP 12: BIND GROUPS (PING-PONG) ===
        //
        // Two bind groups reference the same two buffers with their read/write
        // roles SWAPPED:
        //
        //   Ping: reads buffer[0], writes buffer[1]
        //   Pong: reads buffer[1], writes buffer[0]
        //
        // After each step the active bind group is swapped, so output feeds back
        // in as the next input without reading and writing the same buffer at once.
        let bind_group = [
            // === PING BIND GROUP ===
            context
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bind Group Ping"),
                    layout: &instance_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,  // Binding 0 = READ buffer
                            resource: instance_buffer[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,  // Binding 1 = WRITE buffer
                            resource: instance_buffer[1].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,  // Time uniform (unused)
                            resource: time_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,  // Physics parameters
                            resource: physics_buffer.as_entire_binding(),
                        }
                    ],
                }),
            // === PONG BIND GROUP ===
            // Read/write buffers are swapped relative to Ping.
            context
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bind Group Pong"),
                    layout: &instance_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,  // READ buffer (now buffer[1])
                            resource: instance_buffer[1].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,  // WRITE buffer (now buffer[0])
                            resource: instance_buffer[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,  // Time uniform (same)
                            resource: time_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,  // Physics parameters (same)
                            resource: physics_buffer.as_entire_binding(),
                        }
                    ],
                }),
        ];

        // === STEP 13: SPHERE PIPELINE ===
        //
        // The central sphere needs its own pipeline because it does NOT use
        // instancing (drawn once) and uses different shader entry points
        // (sphere_vs_main, sphere_fs_main).
        let sphere_shader = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Sphere Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });

        // Same layout as the cloth: camera only.
        let sphere_pipeline_layout = context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sphere Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Sphere render pipeline. Key difference: buffers: &[Vertex::desc()] with
        // NO Instance::desc(), because the sphere is static (no instancing).
        let sphere_render_pipeline = context
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Sphere Render Pipeline"),
                layout: Some(&sphere_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &sphere_shader,
                    entry_point: "sphere_vs_main",       // Sphere-specific entry point
                    buffers: &[Vertex::desc()],          // Vertices ONLY, no instances
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &sphere_shader,
                    entry_point: "sphere_fs_main",       // Sphere-specific entry point
                    targets: &[Some(wgpu::ColorTargetState {
                        format: context.format(),
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: context.depth_stencil_format(),
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        // === STEP 14: RETURN THE FULLY INITIALIZED STRUCT ===
        //
        // Everything (buffers, pipelines, state) is now ready.
        Self {
            // Cloth buffers
            vertex_buffer,         // Mini-sphere geometry
            instance_buffer,       // [2] ping-pong buffers
            index_buffer,          // Triangle indices

            // GPU pipelines
            render_pipeline,       // Cloth rendering
            compute_pipeline,      // Physics step

            // Metadata
            num_indices,           // Indices to draw
            num_instances,         // Particle count

            // Camera and timing
            camera,                // Controllable orbit camera
            last_generation: Instant::now(), // Timing timestamp

            // Ping-pong bind groups
            bind_group,            // [2] alternating bind groups

            // Central sphere
            sphere_index_buffer,
            sphere_vertex_buffer,
            num_sphere_indices: sphere_indices.len() as u32,
            sphere_render_pipeline,

            // User settings
            settings: settings.clone(),        // Currently applied settings
            pending_settings: settings,        // Settings edited in the UI
            needs_rebuild: false,              // Rebuild-needed flag
            paused: false,                     // Pause/play state
        }
    }

    /// Rebuilds the whole simulation with the pending settings (grid size,
    /// spacing, etc.). Called when the user clicks "Apply and Restart": create a
    /// fresh app with the new settings, then move its buffers/pipelines over.
    fn rebuild(&mut self, context: &Context) {
        let new_app = Self::create_with_settings(context, self.pending_settings.clone());
        self.vertex_buffer = new_app.vertex_buffer;
        self.instance_buffer = new_app.instance_buffer;
        self.index_buffer = new_app.index_buffer;
        self.num_indices = new_app.num_indices;
        self.num_instances = new_app.num_instances;
        self.bind_group = new_app.bind_group;
        self.sphere_vertex_buffer = new_app.sphere_vertex_buffer;
        self.sphere_index_buffer = new_app.sphere_index_buffer;
        self.num_sphere_indices = new_app.num_sphere_indices;
        self.settings = self.pending_settings.clone();
        self.needs_rebuild = false;
    }

    /// Updates the cloth and sphere colors in place, without rebuilding the
    /// buffers. Regenerates the vertices with the new color and uploads them via
    /// write_buffer (which is why those buffers carry COPY_DST). This is faster
    /// and smoother for the user than a full rebuild.
    fn update_colors(&mut self, context: &Context) {
        let grid_size = round_grid_size(self.settings.grid_size);
        let (new_vertices, _, _, _, _) = generate_grid(
            context,
            grid_size,
            grid_size,
            self.settings.spacing,
            0.5,
            self.settings.point_size,
            self.pending_settings.cloth_color, // New color picked in the UI
        );
        // Upload the recolored vertices into the existing GPU buffer.
        context.queue().write_buffer(
            &self.vertex_buffer,
            0,  // Start of the buffer
            bytemuck::cast_slice(&new_vertices),
        );

        // Same for the central obstacle sphere.
        let (sphere_vertices, _) = create_sphere_vertices(0.4, self.pending_settings.sphere_color);
        context.queue().write_buffer(
            &self.sphere_vertex_buffer,
            0,
            bytemuck::cast_slice(&sphere_vertices),
        );

        // Record the applied colors in the settings.
        self.settings.cloth_color = self.pending_settings.cloth_color;
        self.settings.sphere_color = self.pending_settings.sphere_color;
    }
}

impl App for InstanceApp {
    fn input(&mut self, input: egui::InputState, context: &Context) {
        // Forward user input (mouse, keyboard) to the orbit camera.
        self.camera.input(input, context);
    }

    fn render_gui(&mut self, egui_ctx: &egui::Context, context: &Context) {
        // Control panel: colors, grid size, spacing, etc.
        egui::Window::new("Cloth Settings").show(egui_ctx, |ui| {
            // Pause/play button to stop or resume the simulation.
            if ui.button(if self.paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                self.paused = !self.paused;
            }
            ui.separator();

            // Cloth color picker.
            ui.label("Cloth color:");
            let mut cloth_color = self.pending_settings.cloth_color;
            if ui.color_edit_button_rgb(&mut cloth_color).changed() {
                self.pending_settings.cloth_color = cloth_color;
                self.update_colors(context); // Apply the new color immediately
            }

            // Sphere color picker.
            ui.label("Sphere color:");
            let mut sphere_color = self.pending_settings.sphere_color;
            if ui.color_edit_button_rgb(&mut sphere_color).changed() {
                self.pending_settings.sphere_color = sphere_color;
                self.update_colors(context); // Apply the new color immediately
            }

            ui.separator();
            ui.label("Settings (restart required):");

            // Grid size slider (number of particles per side).
            ui.horizontal(|ui| {
                ui.label("Grid size:");
                let mut grid_val = self.pending_settings.grid_size as i32;
                if ui.add(egui::Slider::new(&mut grid_val, 64..=512).step_by(64.0)).changed() {
                    self.pending_settings.grid_size = grid_val as u32;
                }
            });
            ui.label(format!("  → {} particles", self.pending_settings.grid_size * self.pending_settings.grid_size));

            // Particle spacing slider.
            ui.horizontal(|ui| {
                ui.label("Spacing:");
                ui.add(egui::Slider::new(&mut self.pending_settings.spacing, 0.002..=0.02).step_by(0.001));
            });

            // Visual particle size slider.
            ui.horizontal(|ui| {
                ui.label("Point size:");
                ui.add(egui::Slider::new(&mut self.pending_settings.point_size, 0.001..=0.01).step_by(0.0005));
            });

            ui.separator();

            // Warn when pending settings require a rebuild.
            let settings_changed = self.pending_settings.grid_size != self.settings.grid_size
                || self.pending_settings.spacing != self.settings.spacing
                || self.pending_settings.point_size != self.settings.point_size;

            if settings_changed {
                ui.colored_label(egui::Color32::YELLOW, "⚠️ Pending changes");
                if ui.button("🔄 Apply and Restart").clicked() {
                    self.rebuild(context); // Rebuild the whole simulation
                }
            }

            ui.separator();
            // Total number of simulated particles.
            ui.label(format!("Particles: {}", self.num_instances));
        });
    }

    fn update(&mut self, delta_time: f32, context: &Context) {
        // Called every frame to advance the physics, using a fixed time step for
        // numerical stability.
        if self.paused {
            return;
        }

        let fixed_timestep = TAYME; // Fixed step (e.g. 0.0016 s)
        let mut accumulated_time = delta_time;

        // Run as many fixed steps as needed to catch up with the real elapsed
        // time (more steps when rendering is slower than the simulation rate).
        while accumulated_time >= fixed_timestep {
            // Command encoder for the compute pass.
            let mut encoder = context.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

            {
                // Begin a compute pass to run the physics kernel.
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });

                // Select the physics compute pipeline.
                compute_pass.set_pipeline(&self.compute_pipeline);
                // Bind the current ping (or pong) bind group.
                compute_pass.set_bind_group(0, &self.bind_group[0], &[]);
                // Dispatch one workgroup per WORKGROUP_SIZE particles.
                compute_pass.dispatch_workgroups(self.num_instances / WORKGROUP_SIZE, 1, 1);
            }

            // Submit the work to the GPU.
            context.queue().submit(std::iter::once(encoder.finish()));

            // Ping-pong swap: the buffer just written becomes the next read buffer.
            self.instance_buffer.swap(0, 1);
            self.bind_group.swap(0, 1);

            // Consume one fixed step of accumulated time.
            accumulated_time -= fixed_timestep;

            // Record when this step ran.
            self.last_generation = Instant::now();
        }
    }
    fn render(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Draws the scene each frame.
        // 1. Bind the camera (view/projection matrices).
        render_pass.set_bind_group(0, self.camera.bind_group(), &[]);

        // 2. Draw the cloth (all particles via instancing).
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..)); // Mini-sphere geometry
        render_pass.set_vertex_buffer(1, self.instance_buffer[0].slice(..)); // Particle positions
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);

        // 3. Draw the central obstacle sphere.
        render_pass.set_pipeline(&self.sphere_render_pipeline);
        render_pass.set_vertex_buffer(0, self.sphere_vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.sphere_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_sphere_indices, 0, 0..1);
    }
}

// ============================================================================
// UNIT TESTS (CPU-ONLY)
// ============================================================================
//
// These tests run under `cargo test` on any machine: they exercise only the
// pure-CPU logic and the CPU/GPU struct-layout invariants. They do NOT touch a
// GPU device, surface, or the wgpu pipelines. The physics itself runs in a WGSL
// compute shader and cannot be unit-tested here (see the README "Testing"
// section for what that would require).
#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    // ---- CPU/GPU struct layout invariants ----
    //
    // The CPU uploads these structs to GPU buffers as raw bytes (bytemuck) and
    // the WGSL side reads them back by byte offset, not by field name. A size or
    // offset mismatch is a real, silent bug: the shader would read garbage. These
    // tests pin the layout the WGSL declarations rely on.

    #[test]
    fn vertex_layout_matches_shader() {
        // shader.wgsl VertexInput: position/normal/color are each vec3<f32> read
        // from the vertex buffer at offsets 0, 12, 24 (see Vertex::desc()).
        // The struct is 3 x vec3<f32> = 9 floats = 36 bytes, 4-byte aligned.
        assert_eq!(size_of::<Vertex>(), 36, "Vertex must be 9 f32 = 36 bytes");
        assert_eq!(align_of::<Vertex>(), 4, "Vertex is f32-aligned");
        assert_eq!(offset_of!(Vertex, position), 0);
        assert_eq!(offset_of!(Vertex, normal), 12);
        assert_eq!(offset_of!(Vertex, color), 24);
    }

    #[test]
    fn vertex_desc_offsets_match_struct() {
        // The hand-written byte offsets in Vertex::desc() must match the actual
        // struct offsets, otherwise the GPU reads attributes from wrong locations.
        let desc = Vertex::desc();
        assert_eq!(desc.array_stride, size_of::<Vertex>() as wgpu::BufferAddress);
        assert_eq!(desc.attributes[0].offset, offset_of!(Vertex, position) as u64);
        assert_eq!(desc.attributes[1].offset, offset_of!(Vertex, normal) as u64);
        assert_eq!(desc.attributes[2].offset, offset_of!(Vertex, color) as u64);
    }

    #[test]
    fn instance_layout_matches_shader() {
        // compute.wgsl Instance: position + speed are each vec4<f32> (std430:
        // 16-byte aligned, 16-byte size). On the CPU they are [f32; 4]. So the
        // struct is 32 bytes with speed at offset 16. The vec4 (with padding w)
        // is what keeps the CPU and GPU layouts in agreement.
        assert_eq!(size_of::<Instance>(), 32, "Instance must be 2 x vec4 = 32 bytes");
        assert_eq!(align_of::<Instance>(), 4);
        assert_eq!(offset_of!(Instance, position), 0);
        assert_eq!(offset_of!(Instance, speed), 16, "speed must start at offset 16 (vec4 alignment)");
    }

    #[test]
    fn instance_desc_offsets_match_struct() {
        // Instance::desc() exposes position (loc 3) at offset 0 and speed (loc 4)
        // at offset 12 (= size_of::<[f32;3]>()). Note: this reads the velocity as
        // a Float32x3 starting at byte 12, i.e. the last position float + first
        // two speed floats. The render shader ignores @location(4), so this only
        // needs the stride to match; we still pin the documented offsets.
        let desc = Instance::desc();
        assert_eq!(desc.array_stride, size_of::<Instance>() as wgpu::BufferAddress);
        assert_eq!(desc.attributes[0].offset, 0);
        assert_eq!(desc.attributes[1].offset, size_of::<[f32; 3]>() as u64);
    }

    #[test]
    fn physics_params_layout_matches_shader() {
        // compute.wgsl PhysicsParams: 9 consecutive f32 scalars. On the CPU,
        // #[repr(C)] packs them tightly into 36 bytes. Field ORDER is load-bearing
        // (mapped by offset), so we pin each offset to catch any reordering.
        assert_eq!(size_of::<PhysicsParams>(), 36, "9 f32 = 36 bytes");
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
    }

    #[test]
    fn time_uniform_layout_matches_shader() {
        // compute.wgsl TimeUniform: a single f32.
        assert_eq!(size_of::<TimeUniform>(), 4);
        assert_eq!(offset_of!(TimeUniform, generation_duration), 0);
    }

    // ---- WORKGROUP_SIZE rounding of grid_size ----

    #[test]
    fn round_grid_size_rounds_down_to_multiple() {
        // Exact multiples are unchanged.
        assert_eq!(round_grid_size(256), 256);
        assert_eq!(round_grid_size(512), 512);
        // Non-multiples round DOWN to the nearest multiple of WORKGROUP_SIZE.
        assert_eq!(round_grid_size(300), 256);
        assert_eq!(round_grid_size(511), 256);
        assert_eq!(round_grid_size(513), 512);
    }

    #[test]
