# GPU Cloth Simulation

<p align="center">
  <img src="assets/cloth.svg" alt="GPU cloth draping over a sphere" width="520">
</p>


A real-time cloth simulation that runs its physics entirely on the GPU, written in Rust with [wgpu](https://wgpu.rs/) (WebGPU) and WGSL compute shaders. A square piece of cloth falls under gravity, collides with a sphere and the ground, and settles, with every spring force computed in parallel on the GPU.

Built for the parallel-programming course at ECAM (Brussels).

This is a **learning project**. I built it to get hands-on with two things at once: **parallel programming** (moving a physics solver onto thousands of GPU threads that all run at the same time, which forces you to design around data races, double-buffering and workgroups) and the **Rust** language (ownership, traits and zero-cost abstractions, plus the `wgpu`/WGSL ecosystem for talking to the GPU). The cloth is the excuse; the real goal was the GPU compute model and learning Rust.

## What it demonstrates

- A **mass-spring physical model** solved on the GPU rather than the CPU, so thousands of particles are integrated in parallel every frame.
- A real **GPU compute pipeline** (WGSL compute shader) feeding a separate **render pipeline**, the two communicating through GPU storage buffers.
- A **ping-pong buffer** scheme (read from `ping`, write to `pong`, then swap) to update particle state without read/write conflicts across parallel invocations.
- Interactive tuning of the physics in real time through an [egui](https://github.com/emilk/egui) panel.

## The physics model

Each cloth particle is a point mass linked to its neighbours by three families of springs (Hooke's law, with velocity damping to keep the system stable):

- **Structural** springs (horizontal and vertical neighbours) hold the grid together.
- **Shear** springs (diagonal neighbours) resist in-plane shearing.
- **Bend** springs (two cells apart) resist folding.

On top of the spring forces, the compute shader applies gravity, ground collision, collision against a sphere of configurable radius, friction, and a distance constraint that caps how far a spring can stretch (to avoid the cloth exploding at large time steps).

## Architecture

```
main.rs
  └── sets up the wgpu-bootstrap Runner (window, device/queue, egui, frame loop)
        └── InstanceApp (instances_app.rs)   the whole application
              ├── Compute pipeline  ── compute.wgsl   physics step (spring forces + integration)
              │     ├── instances_ping / instances_pong   particle state (position + velocity)
              │     ├── TimeUniform        time step / duration
              │     └── PhysicsParams      stiffness, damping, mass, rest length, sphere radius...
              │
              └── Render pipeline   ── shader.wgsl
                    ├── draws the cloth grid (instanced from the simulated positions)
                    ├── draws the colliding sphere (icosphere)
                    └── OrbitCamera (view + projection)
```

Each frame: the compute shader integrates one physics step into the `pong` buffer, the buffers are swapped, then the render pipeline draws the cloth and the sphere from the updated positions.

## Tech stack

- **Rust** (edition 2021)
- **wgpu** / **WebGPU** with **WGSL** shaders (one compute shader, one render shader)
- [**wgpu-bootstrap**](https://github.com/qlurkin/wgpu-bootstrap) for window, device and frame-loop boilerplate
- **egui** (real-time parameter UI), **cgmath** (3D math), **bytemuck** (CPU/GPU data layout)

## Getting started

You need a recent [Rust toolchain](https://www.rust-lang.org/tools/install) and a machine with a GPU that supports WebGPU (Vulkan, Metal or DX12).

```bash
git clone https://github.com/DeharengOlivier/<repo>.git
cd <repo>
cargo run --release
```

Use `--release`: the simulation is much smoother with optimizations on.

## Controls

- **Orbit camera**: drag with the mouse to rotate, scroll to zoom.
- **egui panel**: tune the physics live (spring stiffness for structural / shear / bend, damping, mass, rest length, time step, friction, sphere radius) and watch the cloth react.

## Project structure

```
src/
├── main.rs            entry point, configures and launches the Runner
├── instances_app.rs   application: buffers, pipelines, camera, egui, frame update
├── compute.wgsl       GPU physics step (spring forces, integration, collisions)
└── shader.wgsl        GPU rendering (cloth grid + sphere)
```

## Testing

```bash
cargo test
```

`cargo test` runs a small suite of **CPU-only** unit tests (in `src/instances_app.rs`). They need no GPU, window, or surface, so they run anywhere the project compiles, including CI. They cover:

- **CPU/GPU struct-layout invariants.** Using `std::mem::size_of` / `align_of` / `offset_of!`, the tests pin the size and field offsets of `Vertex`, `Instance`, `TimeUniform`, and `PhysicsParams`, and check that the byte offsets declared in `Vertex::desc()` / `Instance::desc()` match the actual struct layout. These structs are uploaded to the GPU as raw bytes (`bytemuck`) and read back by the WGSL shaders **by byte offset, not by field name**, so a size or field-order mismatch is a real, silent bug class. The tests encode the layout the WGSL declarations rely on (for example `Instance` = two `vec4<f32>` = 32 bytes with `speed` at offset 16, and the nine `f32` fields of `PhysicsParams` in their exact order).
- **`WORKGROUP_SIZE` rounding of `grid_size`.** `round_grid_size` rounds a requested side length down to a multiple of `WORKGROUP_SIZE` and clamps it to at least one full workgroup; the tests check the rounding, the clamp (including 0), and that the resulting particle count (`side * side`) is always a whole multiple of `WORKGROUP_SIZE` so the compute dispatch leaves no particle unprocessed.
- **Cloth grid generation.** `generate_instances` (the pure-CPU part of `generate_grid`) is tested for particle count, row-major ordering (`index = row * cols + col`), the documented centering offset, the padding component being zero, and every particle starting at rest.

**What `cargo test` does NOT cover (honest scope):** the physics itself runs in a WGSL compute shader on the GPU and is **not** unit-tested here. Spring forces, integration, collisions, and the distance constraints are validated **visually** by running the simulation. Testing them automatically would require a **headless GPU harness**: create a `wgpu` device without a surface, run the compute pass for a fixed number of steps, read the buffers back, and assert invariants (no NaNs, bounded energy, particles staying within the distance constraint). That harness does not exist yet and would need an actual GPU (or a software adapter such as `llvmpipe`/WARP) available in CI, so it is out of scope for the current `cargo test`.

