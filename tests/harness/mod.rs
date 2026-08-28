//! A wgpu device with no window and no surface, so the compute shader can be
//! run and its output read back inside `cargo test`.
//!
//! Every backend wgpu supports can do this: Metal and Vulkan on real hardware,
//! and Mesa's lavapipe software rasteriser on a CI runner with no GPU at all.

use gpu_cloth_simulation::simulation::{ClothConfig, ClothSimulation, Instance};
use gpu_cloth_simulation::wgpu;
use pollster::FutureExt;

/// A headless device and queue.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// What the adapter reported, so a failure names the backend it happened on.
    pub description: String,
}

impl Gpu {
    /// Acquires a headless adapter, or reports that there is none.
    ///
    /// Returns `None` rather than panicking so a developer with no usable
    /// adapter gets a skipped test with a reason instead of a red suite. CI
    /// installs a software adapter and asserts one is present, so "no adapter"
    /// can never quietly mean "physics untested" there.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Headless Test Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .block_on()
            .ok()?;
        Some(Self {
            device,
            queue,
            description: format!("{:?} / {}", info.backend, info.name),
        })
    }
}

/// Runs a simulation for `steps` fixed time steps and returns the final state.
pub fn simulate(gpu: &Gpu, config: &ClothConfig, steps: u32) -> Vec<Instance> {
    let mut simulation = ClothSimulation::new(&gpu.device, config);
    for _ in 0..steps {
        simulation.step(&gpu.device, &gpu.queue);
    }
    simulation.read_particles(&gpu.device, &gpu.queue)
}

/// The smallest cloth the simulation can build, so a test can step it thousands
/// of times in a second.
pub fn small_config() -> ClothConfig {
    ClothConfig {
        grid_size: 1, // Raised to the one-workgroup minimum.
        spacing: 0.006,
        initial_height: 0.5,
        ..ClothConfig::default()
    }
}

/// The distance between two particles.
pub fn distance(a: &Instance, b: &Instance) -> f32 {
    let (dx, dy, dz) = (
        a.position[0] - b.position[0],
        a.position[1] - b.position[1],
        a.position[2] - b.position[2],
    );
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// The distance of a particle from the origin, where the obstacle sphere sits.
pub fn radius(particle: &Instance) -> f32 {
    let p = particle.position;
    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
}

/// The larger of two values, or NaN if either is NaN.
///
/// `f32::max` returns the other operand when one of them is NaN, so folding a
/// sheet that has already exploded with it reports a healthy number and the
/// assertion built on it passes. Every maximum taken over particle state goes
/// through this instead, so a NaN reaches the assertion and fails it.
pub fn larger(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a > b {
        a
    } else {
        b
    }
}

/// The largest stretch factor of any spring in the sheet, structural or shear.
pub fn worst_stretch(state: &[Instance], side: usize, spacing: f32) -> f32 {
    let diagonal = spacing * std::f32::consts::SQRT_2;
    let mut worst = 0.0f32;
    for row in 0..side {
        for col in 0..side {
            let here = row * side + col;
            if col + 1 < side {
                worst = larger(worst, distance(&state[here], &state[here + 1]) / spacing);
            }
            if row + 1 < side {
                worst = larger(worst, distance(&state[here], &state[here + side]) / spacing);
            }
            if col + 1 < side && row + 1 < side {
                worst = larger(
                    worst,
                    distance(&state[here], &state[here + side + 1]) / diagonal,
                );
            }
        }
    }
    worst
}

/// The speed of the fastest particle, or NaN if any of them is NaN.
pub fn top_speed(state: &[Instance]) -> f32 {
    state
        .iter()
        .map(|p| (p.speed[0].powi(2) + p.speed[1].powi(2) + p.speed[2].powi(2)).sqrt())
        .fold(0.0f32, larger)
}

/// Total kinetic energy, up to the constant factor of mass / 2.
pub fn kinetic_energy(state: &[Instance]) -> f32 {
    state
        .iter()
        .map(|p| p.speed[0].powi(2) + p.speed[1].powi(2) + p.speed[2].powi(2))
        .sum()
}

/// Mean height of the sheet.
pub fn mean_height(state: &[Instance]) -> f32 {
    state.iter().map(|p| p.position[1]).sum::<f32>() / state.len() as f32
}

/// A device, or `None` after printing why the calling test is skipping.
pub fn gpu_or_skip(test_name: &str) -> Option<Gpu> {
    match Gpu::new() {
        Some(gpu) => {
            eprintln!("{test_name}: running on {}", gpu.description);
            Some(gpu)
        }
        None => {
            eprintln!("{test_name}: SKIPPED, no wgpu adapter on this machine");
            None
        }
    }
}
