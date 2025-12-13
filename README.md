# GPU Cloth Simulation

<p align="center">
  <img src="assets/cloth.svg" alt="GPU cloth draping over a sphere" width="520">
</p>


A real-time cloth simulation that runs its physics entirely on the GPU, written in Rust with [wgpu](https://wgpu.rs/) (WebGPU) and WGSL compute shaders. A square piece of cloth falls under gravity, collides with a sphere and the ground, and settles, with every spring force computed in parallel on the GPU.

Built for the parallel-programming course at ECAM (Brussels).

This is a **learning project**. I built it to get hands-on with two things at once: **parallel programming** (moving a physics solver onto thousands of GPU threads that all run at the same time, which forces you to design around data races, double-buffering and workgroups) and the **Rust** language (ownership, traits and zero-cost abstractions, plus the `wgpu`/WGSL ecosystem for talking to the GPU). The cloth is the excuse; the real goal was the GPU compute model and learning Rust.

