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
