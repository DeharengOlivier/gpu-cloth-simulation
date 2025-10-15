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
