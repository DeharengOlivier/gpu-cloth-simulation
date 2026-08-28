# GPU Cloth Simulation

<p align="center">
  <img src="assets/cloth.svg" alt="GPU cloth draping over a sphere" width="520">
</p>

A real-time cloth simulation whose physics runs entirely on the GPU, written in
Rust with [wgpu](https://wgpu.rs/) (WebGPU) and WGSL compute shaders. A square
sheet falls under gravity, collides with a sphere and the ground, and settles.
Every spring force is computed in parallel on the GPU.

Built for the parallel-programming course at ECAM (Brussels).

This is a **learning project**, built to get hands-on with two things at once:
**parallel programming**, moving a physics solver onto thousands of GPU threads
that all run at the same time, which forces the design around data races,
double-buffering and workgroups; and **Rust**, along with the `wgpu`/WGSL
ecosystem for talking to the GPU. The cloth is the excuse.

## What it demonstrates

- A **mass-spring model** solved on the GPU rather than the CPU, so hundreds of
  thousands of particles are integrated in parallel every frame.
- A **compute pipeline** (WGSL) feeding a separate **render pipeline**, the two
  sharing GPU storage buffers with no round trip through the CPU.
- A **ping-pong buffer** scheme (read `ping`, write `pong`, swap) that updates
  particle state without read/write conflicts across parallel invocations.
- The physics **asserted rather than watched**: the compute shader runs on a
  headless device inside `cargo test`, and the particle buffer is read back and
  checked against the invariants the simulation claims.

## The physics model

Each particle is a point mass linked to its neighbours by three families of
springs (Hooke's law, with velocity damping):

- **Structural** springs, to the four direct neighbours, hold the grid together.
- **Shear** springs, to the four diagonal neighbours, resist in-plane shearing.
- **Bend** springs, two cells apart, resist folding.

On top of the spring forces the shader applies gravity, collision with the
ground plane and with a sphere, and Coulomb friction against that sphere.

Each step is then followed by a **positional constraint pass** that pulls back
any spring stretched past `MAX_SPRING_STRETCH` (1.5x its rest length). It is a
safeguard, not a shaping tool, and the value is bracketed by two measurements
taken with the constraint disabled across the whole range the interface offers:

- a sheet **at rest** stretches up to 1.365x under its own weight, so a cap
  below that would be unsatisfiable, firing on every step forever and fighting
  gravity instead of catching a divergence;
- the **transient**, as the falling sheet snaps taut on the sphere, reaches
  3.04x, which is not cloth, so there is real work for the cap to do.

Two relaxation passes run per step: the transient peak lands at 1.59x with two
and 1.81x with one.

## Architecture

The dependency arrow points inward. `simulation` needs a `wgpu::Device` and
nothing else, which is what makes it testable without a window.

```
main.rs               configures and launches the wgpu-bootstrap runner
  └── app.rs          the interactive program: window, camera, egui, rendering
        ├── timestep.rs    elapsed frame time -> whole physics steps (no GPU)
        └── simulation.rs  the cloth itself
              ├── compute.wgsl   integration pass, then the constraint pass
              ├── instances_ping / instances_pong   particle state, ping-ponged
              └── PhysicsParams  stiffness, damping, mass, rest length, friction,
                                 sphere radius, stretch cap, grid side
headless.rs           a device with no window, for tests and benchmarks
shader.wgsl           rendering: the cloth grid, instanced, and the sphere
```

Per frame: `timestep` says how many fixed steps the elapsed time paid for, each
step dispatches the integration pass and then the constraint passes, and the
render pipeline draws straight from the buffer the physics just wrote.

## Getting started

You need a recent [Rust toolchain](https://www.rust-lang.org/tools/install) and
a GPU that supports WebGPU (Vulkan, Metal or DX12).

```bash
git clone https://github.com/DeharengOlivier/gpu-cloth-simulation.git
cd gpu-cloth-simulation
cargo run --release
```

Use `--release`. A debug build runs the simulation far below real time.

## Controls

- **Camera**: drag to orbit, scroll to zoom.
- **Panel, applied immediately**: pause and resume, the cloth and sphere
  colours, the three spring stiffnesses, the damping, the particle mass and the
  friction. These are uniforms, so they are rewritten between steps and a draped
  sheet carries on rather than being dropped and re-fallen.
- **Panel, needing a restart**: the grid side (64 to 512), the spacing between
  particles and the drawn point size, since each changes the set of particles.
  The panel offers the restart once one of them has moved.

It also reports the particle count, the steps the last frame ran against the
10.4 real time asks for, and a warning once the machine has fallen behind.

## Performance

`cargo run --release --example throughput` measures it. Simulated time keeps up
with real time at 625 steps per second, and one 60 fps frame needs 10.4 steps.
On an Apple M5 Pro:

| grid | particles | steps/s | share of a 60 fps frame |
| ---: | --------: | ------: | ----------------------: |
|   64 |     4 096 |   6 809 |                    9.2% |
|  128 |    16 384 |   5 894 |                   10.6% |
|  256 |    65 536 |   3 470 |                   18.0% |
|  512 |   262 144 |   1 875 |                   33.3% |

Regenerate rather than edit these, and say which machine they came from.

## Testing

```bash
cargo test                              # everything except the two heavy tests
cargo test --test physics -- --ignored  # the two, on the largest supported grid
```

**CPU tests**, needing no device, cover the layout the CPU and the GPU have to
agree on, byte for byte, since `bytemuck` uploads these structs as raw bytes and
the WGSL reads them back by offset rather than by name: the size and every field
offset of `Vertex`, `Instance` and `PhysicsParams`, checked against the WGSL
declarations parsed out of the shader source. They also cover grid resolution,
particle generation, and the frame clock in `timestep`.

**Physics tests** run the real `compute.wgsl` on a headless device and read the
particle buffer back.

One of them is a comparison against a known answer rather than an invariant.
Before the sheet reaches the sphere it is flat and every particle moves
identically, so no spring is stretched: what is left is gravity and air drag
under semi-implicit Euler, which has an exact closed form. The GPU matches it
to a relative error of 3e-7, which is f32 round-off, and the test holds at 1e-5.
That single comparison pins gravity, the drag coefficient, the mass, the time
step and the integration scheme at once.

The rest are invariants. They assert that no particle becomes NaN, that the sheet
falls, that nothing passes through the ground or ends up inside the sphere, that
nothing is flung above the release height or outruns ten free falls from it,
that the sheet stays bounded horizontally, that every particle is stepped
including in a partial workgroup, that two identical runs agree, that the grid
side asked for is the one simulated, that the friction coefficient reaches the
shader, that the cloth settles, that a settled sheet stays under the stretch
cap, and that the cap measurably clips the transient against an unconstrained
run.

A machine with no adapter skips them with a message rather than failing. CI
installs Mesa's lavapipe, a Vulkan implementation that runs on the CPU, and one
test asserts an adapter is present whenever `CI` is set, so a missing rasteriser
fails the build instead of quietly turning the suite green.

The two tests that need 262,144 particles for thousands of steps are ignored by
default and run nightly: seconds on a real GPU, the whole build on a software
one.

## Limitations

- **Self-collision is not handled.** The cloth passes through itself. GPU
  spatial hashing for the broad phase is the natural next step.
- **The sheet is square.** Non-square cloths are not supported, though the
  dispatch no longer constrains the resolution.
- **The workgroup size is fixed at 256 and untuned.** The best value is
  hardware-dependent, and a 2D workgroup would map more naturally onto a 2D
  grid. The benchmark above is the place to settle that with numbers.
- **The sphere radius is not in the panel.** Every other physics parameter is;
  this one changes the obstacle mesh as well as the uniform, so it needs the
  restart path rather than the live one.
- **The constraint is Jacobi, not Gauss-Seidel.** Each invocation applies half
  of each spring's excess against neighbours that move in the same pass, so a
  fixed iteration count leaves a residue: 1.59x against a 1.5x cap at the
  heaviest supported setting.
- **Only the free-fall phase has a reference solution.** Once the sheet is in
  contact and the springs are loaded there is no closed form to check against,
  so from that point on the tests assert invariants, which catch an exploding
  or frozen sheet rather than a subtly wrong drape.
- **One coefficient plays two roles.** `damping` is both the spring damping and
  the global air drag, so tuning it moves two independent things at once.

## License

Released under the MIT License. See [LICENSE](LICENSE).
