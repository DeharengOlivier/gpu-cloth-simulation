//! A wgpu device with no window and no surface.
//!
//! The simulation needs a `Device` and a `Queue` and nothing else: no surface,
//! no swapchain, no event loop. Exposing that here is what lets the physics be
//! run and read back from a test, a benchmark, or any program that wants the
//! cloth without a window.
//!
//! Every backend wgpu supports can do this: Metal and Vulkan on real hardware,
//! and Mesa's lavapipe software rasteriser on a machine with no GPU at all.

use pollster::FutureExt;

use crate::wgpu;

/// A headless device and queue, with the adapter that produced them named.
pub struct Gpu {
    /// The device every GPU resource is created from.
    pub device: wgpu::Device,
    /// The queue every command buffer is submitted to.
    pub queue: wgpu::Queue,
    /// Backend and adapter name, so a failure says where it happened.
    pub description: String,
}

impl Gpu {
    /// Acquires a headless adapter and device.
    ///
    /// Returns `None` rather than panicking when the machine has no usable
    /// adapter, so a caller can skip with a reason instead of dying. Blocks
    /// until the adapter and device are ready.
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
                    label: Some("Headless Device"),
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
