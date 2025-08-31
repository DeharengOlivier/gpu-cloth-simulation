// Module holding the simulation application.
mod instances_app;

use std::sync::Arc;

use crate::instances_app::InstanceApp;
use wgpu_bootstrap::{egui, Runner};

fn main() {
    // The Runner (from wgpu_bootstrap) owns all the boilerplate needed to run a
    // GPU application, so this file stays small:
    //
    // 1. Window creation (winit)
    // 2. WebGPU/wgpu initialization (device, queue, surface)
    // 3. The main render/update loop
    // 4. Event handling (mouse, keyboard, resize)
    // 5. egui integration (the parameter UI)
    // 6. Timing (delta_time, FPS)

    let mut runner = Runner::new(
        "GPU Cloth Simulation",

        // Initial window width in pixels.
        900,

        // Initial window height in pixels.
        700,

        // egui background color (light grey). Affects the UI only, not the 3D scene.
        egui::Color32::from_rgb(245, 245, 245),

        // MSAA sample count (anti-aliasing). Higher = smoother edges but more cost.
        32,

        // Present mode (0 = VSync, 1 = immediate). VSync caps to the display refresh rate.
        0,

        // App factory closure: the Runner calls it once at startup with the GPU
        // Context (device, queue, surface format) and stores the resulting app in
        // an Arc (thread-safe shared pointer).
        Box::new(|context| Arc::new(InstanceApp::new(context))),
