//! Real-time cloth simulation whose physics runs entirely in a WGSL compute
//! shader.
//!
//! Two layers, and the dependency arrow points inward:
//!
//! - [`simulation`] is the cloth: particle buffers, the compute pipeline, one
//!   physics step. It needs a `wgpu::Device` and nothing else, so its invariants
//!   can be asserted on a headless device rather than judged by eye.
//! - [`app`] is the interactive program built on top: window, camera, egui panel
//!   and the render pipelines. It depends on `simulation`; `simulation` knows
//!   nothing about it.

pub mod app;
pub mod simulation;

/// The wgpu the whole crate is built against, re-exported so callers and tests
/// use exactly this one rather than a second copy resolved on their own.
pub use wgpu_bootstrap::wgpu;
