//! Entry point: configure the window and hand control to the frame loop.

use std::sync::Arc;

use gpu_cloth_simulation::app::InstanceApp;
use wgpu_bootstrap::{egui, Runner};

/// Initial window size, in pixels.
const WINDOW_WIDTH: u32 = 900;
const WINDOW_HEIGHT: u32 = 700;

/// Bit depth of the depth buffer. The scene is depth-tested, so it needs one;
/// 32 bits keeps the small inter-particle distances distinguishable.
const DEPTH_BUFFER_BITS: u8 = 32;

/// Bit depth of the stencil buffer. Nothing here stencils, so none is requested.
const STENCIL_BUFFER_BITS: u8 = 0;

fn main() {
    // The Runner owns the boilerplate: window creation, wgpu initialisation,
    // the frame loop, event handling, egui integration and frame timing.
    let mut runner = Runner::new(
        "GPU Cloth Simulation",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        egui::Color32::from_rgb(245, 245, 245), // egui panel background
        DEPTH_BUFFER_BITS,
        STENCIL_BUFFER_BITS,
        // Called once at startup with the GPU context.
        Box::new(|context| Arc::new(InstanceApp::new(context))),
    );

    // Does not return until the window closes.
    runner.run();
}
