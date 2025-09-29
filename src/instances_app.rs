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
