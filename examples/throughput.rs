//! How many physics steps per second the cloth sustains, per grid size.
//!
//! Run with `cargo run --release --example throughput`. The numbers it prints
//! are the ones quoted in the README; regenerate them rather than editing them
//! by hand, and say which machine they came from.
//!
//! What is measured is the whole step as the application calls it: the
//! integration dispatch plus the constraint passes, submitted and waited on. The
//! wait is the point. Without it the loop would time how fast the CPU can record
//! command buffers, which is not what limits the frame rate.

use std::time::Instant;

use gpu_cloth_simulation::headless::Gpu;
use gpu_cloth_simulation::simulation::{ClothConfig, ClothSimulation, FIXED_TIME_STEP_SECONDS};
use gpu_cloth_simulation::wgpu;

/// Steps run before timing starts, to pay for shader compilation and the first
/// allocations once rather than charging them to the measurement.
const WARMUP_STEPS: u32 = 200;

/// Steps timed per grid size.
const TIMED_STEPS: u32 = 2_000;

/// Frames per second the simulation is expected to keep up with.
const TARGET_FRAME_RATE: f64 = 60.0;

fn main() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no wgpu adapter on this machine: nothing to measure");
        std::process::exit(1);
    };
    println!("adapter: {}", gpu.description);
    println!(
        "step: {FIXED_TIME_STEP_SECONDS} s, so {:.0} steps/s keeps up with real time",
        1.0 / f64::from(FIXED_TIME_STEP_SECONDS)
    );
    println!();
    println!("  grid   particles   steps/s   real-time budget at 60 fps");
    println!("  ----   ---------   -------   --------------------------");

    for grid_size in [64u32, 128, 256, 512] {
        let config = ClothConfig {
            grid_size,
            ..ClothConfig::default()
        };
        let mut simulation = ClothSimulation::new(&gpu.device, &config);

        for _ in 0..WARMUP_STEPS {
            simulation.step(&gpu.device, &gpu.queue);
        }
        gpu.device.poll(wgpu::Maintain::Wait);

        let started = Instant::now();
        for _ in 0..TIMED_STEPS {
            simulation.step(&gpu.device, &gpu.queue);
        }
        gpu.device.poll(wgpu::Maintain::Wait);
        let elapsed = started.elapsed().as_secs_f64();

        let steps_per_second = f64::from(TIMED_STEPS) / elapsed;
        // Steps one frame must run to keep simulated time level with real time.
        let steps_per_frame = 1.0 / (f64::from(FIXED_TIME_STEP_SECONDS) * TARGET_FRAME_RATE);
        let used = steps_per_frame / steps_per_second * TARGET_FRAME_RATE * 100.0;
        println!(
            "  {grid_size:>4}   {:>9}   {steps_per_second:>7.0}   {used:>5.1}% of the frame",
            simulation.particle_count(),
        );
    }
}
