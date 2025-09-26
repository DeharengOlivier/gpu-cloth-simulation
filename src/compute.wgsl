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
    }
}






// Main physics step. One invocation per particle: gather spring forces from its
// grid neighbours, add gravity/damping, resolve collisions, integrate, then apply
// the distance constraints. Reads from instances_ping, writes to instances_pong.
@compute @workgroup_size(WORKGROUP_SIZE)
fn computeMain(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    var instance = instances_ping[index];

    // Maximum allowed stretch factor for the distance constraint (see end of function).
    let max_stretch = 100.0; // Allow 10% stretch

    // Recover the particle's (row, col) in the NxN grid. The grid is square, so
    // its side length is sqrt(total particle count).
    let grid_size = u32(sqrt(f32(arrayLength(&instances_ping))));
    let row = index / grid_size;
    let col = index % grid_size;

    // Current world-space state of this particle, plus the force accumulator.
    let pos = instance.position.xyz;
    let speed = instance.speed.xyz;
    var total_force = vec3<f32>(0.0, 0.0, 0.0);


    // --- Spring forces (Hooke's law) ---
    // Structural springs connect each particle to its 4 direct neighbours
    // (left, right, up, down). Boundary checks skip neighbours off the grid edge.
    // Left neighbour
    if (col > 0) {
        let left_index = index - 1;
        let left_pos = instances_ping[left_index].position.xyz;
        let left_speed = instances_ping[left_index].speed.xyz;
        total_force += calculate_spring_force(pos, left_pos, speed, left_speed, physics.rest_length, physics.structural_k, physics.damping);  
    }

    // Right neighbour
    if (col < grid_size - 1) {
        let right_index = index + 1;
        let right_pos = instances_ping[right_index].position.xyz;
        let right_speed = instances_ping[right_index].speed.xyz;
        total_force += calculate_spring_force(pos, right_pos, speed, right_speed, physics.rest_length, physics.structural_k, physics.damping);
    }

    // Top neighbour
    if (row > 0) {
        let up_index = index - grid_size;
        let up_pos = instances_ping[up_index].position.xyz;
        let up_speed = instances_ping[up_index].speed.xyz;
        total_force += calculate_spring_force(pos, up_pos, speed, up_speed, physics.rest_length, physics.structural_k, physics.damping);
    }

    // Bottom neighbour
    if (row < grid_size - 1) {
        let down_index = index + grid_size;
        let down_pos = instances_ping[down_index].position.xyz;
        let down_speed = instances_ping[down_index].speed.xyz;
        total_force += calculate_spring_force(pos, down_pos, speed, down_speed, physics.rest_length, physics.structural_k, physics.damping);
    }


    // Shear springs connect each particle to its 4 diagonal neighbours and resist
    // in-plane shearing. Their rest length is rest_length * sqrt(2) (the cell diagonal).
    // Top-left diagonal
    if (row > 0 && col > 0) {
        let diag_index = index - grid_size - 1;
        let diag_pos = instances_ping[diag_index].position.xyz;
        let diag_speed = instances_ping[diag_index].speed.xyz;
        total_force += calculate_spring_force(pos, diag_pos, speed, diag_speed, physics.rest_length * sqrt_of_two, physics.shear_k, physics.damping);
    }

    // Top-right diagonal
    if (row > 0 && col < grid_size - 1) {
        let diag_index = index - grid_size + 1;
        let diag_pos = instances_ping[diag_index].position.xyz;
        let diag_speed = instances_ping[diag_index].speed.xyz;
        total_force += calculate_spring_force(pos, diag_pos, speed, diag_speed, physics.rest_length * sqrt_of_two, physics.shear_k, physics.damping);
    }

    // Bottom-left diagonal
    if (row < grid_size - 1 && col > 0) {
        let diag_index = index + grid_size - 1;
        let diag_pos = instances_ping[diag_index].position.xyz;
        let diag_speed = instances_ping[diag_index].speed.xyz;
        total_force += calculate_spring_force(pos, diag_pos, speed, diag_speed, physics.rest_length * sqrt_of_two, physics.shear_k, physics.damping);
    }

    // Bottom-right diagonal
    if (row < grid_size - 1 && col < grid_size - 1) {
        let diag_index = index + grid_size + 1;
        let diag_pos = instances_ping[diag_index].position.xyz;
        let diag_speed = instances_ping[diag_index].speed.xyz;
        total_force += calculate_spring_force(pos, diag_pos, speed, diag_speed, physics.rest_length * sqrt_of_two, physics.shear_k, physics.damping);
    }

    // Bend springs connect each particle to the neighbour two cells away
    // (rest length rest_length * 2). They resist folding and add stiffness so the
    // sheet behaves like cloth rather than a loose net.
    // Two cells to the left
    if (col > 1) {
        let bend_index = index - 2;
        let bend_pos = instances_ping[bend_index].position.xyz;
        let bend_speed = instances_ping[bend_index].speed.xyz;
        total_force += calculate_spring_force(pos, bend_pos, speed, bend_speed, physics.rest_length * 2.0, physics.bend_k, physics.damping);
    }

    // Two cells to the right
    if (col < grid_size - 2) {
        let bend_index = index + 2;
        let bend_pos = instances_ping[bend_index].position.xyz;
        let bend_speed = instances_ping[bend_index].speed.xyz;
        total_force += calculate_spring_force(pos, bend_pos, speed, bend_speed, physics.rest_length * 2.0, physics.bend_k, physics.damping);
    }

    // Two cells up
    if (row > 1) {
        let bend_index = index - (grid_size * 2);
        let bend_pos = instances_ping[bend_index].position.xyz;
        let bend_speed = instances_ping[bend_index].speed.xyz;
        total_force += calculate_spring_force(pos, bend_pos, speed, bend_speed, physics.rest_length * 2.0, physics.bend_k, physics.damping);
    }

    // Two cells down
    if (row < grid_size - 2) {
        let bend_index = index + (grid_size * 2);
        let bend_pos = instances_ping[bend_index].position.xyz;
        let bend_speed = instances_ping[bend_index].speed.xyz;
        total_force += calculate_spring_force(pos, bend_pos, speed, bend_speed, physics.rest_length * 2.0, physics.bend_k, physics.damping);
    }




    // Global velocity damping (air drag): bleeds energy out of the whole system.
    let damping_force = -physics.damping * instance.speed.xyz;
    total_force += damping_force;

    // Gravity (weight = mass * g, applied on the y axis).
    total_force += vec3<f32>(0.0, GRAVITY * physics.mass, 0.0);


    // --- Collision with the central sphere (centred at the origin) ---
    // If the particle is inside the sphere, project it back onto the surface and
    // apply Coulomb friction tangent to the surface.
    let distance = length(instance.position.xyz);
    let radius = physics.sphere_radius;

    if (distance < radius) {
        let normal = normalize(instance.position.xyz);

        // Snap the particle back onto the sphere surface along the outward normal.
        instance.position.x = normal.x * radius;
        instance.position.y = normal.y * radius;
        instance.position.z = normal.z * radius;


        // Coulomb friction: Ff = -min(|F_t|, cf * |F_n|) * t_hat, where F is split
        // into normal (F_n) and tangential (F_t) components relative to the surface.
        let Ro = total_force;
        let In = normal;

        // Normal component of the accumulated force.
        let Ro_n_magnitude = dot(Ro, In);
        let Ro_n = In * Ro_n_magnitude;

        // Tangential component (what friction opposes).
        let Ro_t = Ro - Ro_n;
        let Ro_t_magnitude = length(Ro_t);

        // Only apply friction when there is a non-negligible tangential force.
        if (Ro_t_magnitude > 0.0001) {
            let It = Ro_t / Ro_t_magnitude;

            // Friction coefficient (kept local; not driven by PhysicsParams.friction yet).
            let cf = 0.9;

            // Friction force opposes the tangential motion, capped by cf * |F_n|.
            let friction_magnitude = min(Ro_t_magnitude, cf * abs(Ro_n_magnitude));
            let friction_force = -friction_magnitude * It;

            total_force += friction_force;
        }

        // Reflect the velocity about the surface normal and damp it (inelastic bounce).
        let damping = 0.5;
        let dot_product = dot(instance.speed.xyz, normal);
        instance.speed.x = (instance.speed.x - 2.0 * dot_product * normal.x) * damping;
        instance.speed.y = (instance.speed.y - 2.0 * dot_product * normal.y) * damping;
        instance.speed.z = (instance.speed.z - 2.0 * dot_product * normal.z) * damping;
    }




    // --- Ground collision ---
    // Clamp the particle to the ground plane and damp its downward velocity.
    if (instance.position.y < GROUND) {
        instance.position.y = GROUND;
        let ground_damping = 0.2;
        instance.speed.y = -instance.speed.y * ground_damping;
    }

    // Semi-implicit (symplectic) Euler integration: update velocity from the net
    // force, then advance position using the new velocity.
    let acceleration = total_force / physics.mass;
    instance.speed.x += acceleration.x * physics.dt;
    instance.speed.y += acceleration.y * physics.dt;
    instance.speed.z += acceleration.z * physics.dt;

    // Position update.
    instance.position.x += instance.speed.x * physics.dt;
    instance.position.y += instance.speed.y * physics.dt;
    instance.position.z += instance.speed.z * physics.dt;

    // --- Distance constraints with neighbours ---
    // After integration, hard-cap how far each spring may stretch. NOTE: pos2 is
    // read from instances_ping (this frame's input) and the correction to pos2 is
    // discarded; only this particle's position (pos1) is written back.
    // Left neighbour
    if (col > 0) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index - 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;

    }
    // Right neighbour
    if (col < grid_size - 1) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index + 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
    // Top neighbour
    if (row > 0) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index - grid_size].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
    // Bottom neighbour
    if (row < grid_size - 1) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index + grid_size].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
    // Top-left diagonal neighbour
    if (row > 0 && col > 0) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index - grid_size - 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length * sqrt_of_two, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
    // Top-right diagonal neighbour
    if (row > 0 && col < grid_size - 1) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index - grid_size + 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length * sqrt_of_two, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
    // Bottom-left diagonal neighbour
    if (row < grid_size - 1 && col > 0) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index + grid_size - 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length * sqrt_of_two, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
