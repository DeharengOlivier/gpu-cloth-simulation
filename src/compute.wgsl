// One cloth particle's state. position.xyz / speed.xyz carry the data;
// the w component is padding so the struct matches the std430 16-byte
// alignment expected by the GPU (the host-side Instance uses [f32; 4] too).
struct Instance {
    position: vec4<f32>,
    speed: vec4<f32>,
};

// Time uniform. Declared to match the bind group layout (binding 2) but not
// read by the current simulation step.
struct TimeUniform {
    generation_duration: f32,
};

// Simulation parameters shared by every invocation. Field ORDER must match the
// host-side PhysicsParams struct, because the host/GPU mapping is by byte
// offset, not by name (bytemuck, #[repr(C)]).
struct PhysicsParams {
    structural_k: f32,
    shear_k: f32,
    bend_k: f32,
    damping: f32,
    mass: f32,
    rest_length: f32,
    dt: f32,
    friction: f32,
    sphere_radius: f32,
};

// Bind group 0 for the compute pass. instances_ping is read, instances_pong is
// written (ping-pong scheme: the host swaps which physical buffer is bound here
// after every step, so this frame's output becomes next frame's input).
@group(0) @binding(0) var<storage, read_write> instances_ping: array<Instance>;
@group(0) @binding(1) var<storage, read_write> instances_pong: array<Instance>;
@group(0) @binding(2) var<uniform> time: TimeUniform;
@group(0) @binding(3) var<uniform> physics: PhysicsParams;





// Downward gravity acceleration and the y-coordinate of the ground plane.
const GRAVITY: f32 = -0.3;      // -0.5 also works; lower magnitude = slower fall
const GROUND: f32 = -1.0;
// Precomputed sqrt(2): diagonal (shear) springs have a rest length of
// rest_length * sqrt(2) because the diagonal of a unit grid cell is sqrt(2).
const sqrt_of_two: f32 = 1.41421356237309504880168872420969807856967187537694807317667973799073247846210703885038753432764157273501384623;

// Hooke's law spring with velocity damping: F = -k * (length - rest_length) along
// the spring axis, plus a damping term proportional to the relative velocity along
// that same axis. Damping bleeds off energy so the cloth settles instead of
// oscillating forever. Returns the force exerted on pos1 by the spring to pos2.
fn calculate_spring_force(pos1: vec3<f32>, pos2: vec3<f32>, vel1: vec3<f32>, vel2: vec3<f32>, rest_length: f32, k: f32,  damping: f32) -> vec3<f32> {
    let delta = pos2 - pos1;
    let velocity_delta = vel2 - vel1;
    let current_length = length(delta);

    // Guard against division by zero when two particles coincide.
    if (current_length < 0.0001) {
        return vec3<f32>(0.0);
    }

    let direction = delta / current_length;

    // Elastic restoring force.
    let spring_force = k * (current_length - rest_length) * direction;
    // Damping force (F = -c * v projected on the spring axis) to suppress oscillation.
    let damping_force = damping * dot(velocity_delta, direction) * direction;

    return spring_force + damping_force;
}


// Positional (Jakobsen-style) constraint: if a spring is stretched beyond
// rest_length * max_stretch, push both endpoints back along the axis (half the
// correction each). This is a hard cap applied after integration to stop the
// cloth from exploding when forces are large relative to the time step.
fn enforce_distance_constraint(pos1: ptr<function, vec3<f32>>, pos2: ptr<function, vec3<f32>>, rest_length: f32, max_stretch: f32) {
    let delta = *pos2 - *pos1;
    let current_length = length(delta);

    if current_length > rest_length * max_stretch {
        let correction = delta * (1.0 - (rest_length * max_stretch) / current_length);
        *pos1 += correction * 0.5;
        *pos2 -= correction * 0.5;
