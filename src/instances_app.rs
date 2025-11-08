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

