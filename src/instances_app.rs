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
