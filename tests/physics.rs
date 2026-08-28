//! What the compute shader is supposed to guarantee, asserted rather than watched.
//!
//! The README used to say the physics was "validated visually by running the
//! simulation", and that a headless harness "does not exist yet". It does now:
//! each test below runs the real `compute.wgsl` on a real device with no window,
//! reads the particle buffer back, and checks an invariant the simulation claims.

mod harness;

use gpu_cloth_simulation::simulation::{
    ClothConfig, ClothSimulation, CONSTRAINT_ITERATIONS, MAX_SPRING_STRETCH, WORKGROUP_SIZE,
};

/// The cloth is released from this height, so nothing may ever be above it.
const RELEASE_HEIGHT: f32 = 0.5;
/// The ground plane the shader clamps to.
const GROUND: f32 = -1.0;

#[test]
fn an_adapter_is_available_where_it_must_be() {
    // On CI a software adapter is installed on purpose, so a missing one there
    // would mean every physics test below silently skipped.
    let available = harness::Gpu::new().is_some();
    if std::env::var("CI").is_ok() {
        assert!(
            available,
            "CI must provide a wgpu adapter, even a software one"
        );
    } else if !available {
        eprintln!("no wgpu adapter on this machine: the physics tests will skip");
    }
}

#[test]
fn no_particle_ever_becomes_nan() {
    // The failure mode of an unstable mass-spring solver: one NaN spreads
    // through the springs and the whole sheet disappears in a frame or two.
    let Some(gpu) = harness::gpu_or_skip("no_particle_ever_becomes_nan") else {
        return;
    };
    let particles = harness::simulate(&gpu, &harness::small_config(), 2_000);
    for (index, particle) in particles.iter().enumerate() {
        for value in particle.position.iter().chain(particle.speed.iter()) {
            assert!(
                value.is_finite(),
                "particle {index} holds a non-finite value after 2000 steps: {particle:?}"
            );
        }
    }
}

#[test]
fn the_cloth_falls() {
    // The most basic claim the simulation makes: gravity acts.
    let Some(gpu) = harness::gpu_or_skip("the_cloth_falls") else {
        return;
    };
    let config = harness::small_config();
    let mut simulation = ClothSimulation::new(&gpu.device, &config);
    let before = simulation.read_particles(&gpu.device, &gpu.queue);
    for _ in 0..500 {
        simulation.step(&gpu.device, &gpu.queue);
    }
    let after = simulation.read_particles(&gpu.device, &gpu.queue);

    assert!(
        harness::mean_height(&after) < harness::mean_height(&before),
        "mean height went from {} to {}",
        harness::mean_height(&before),
        harness::mean_height(&after)
    );
}

#[test]
fn a_relaxed_sheet_falls_exactly_as_the_equations_say_it_should() {
    // Every other test here asserts an invariant: bounded, finite, downward. An
    // invariant catches a sheet that explodes or freezes, not one that falls at
    // the wrong rate. This is the one comparison against a known answer.
    //
    // Before the sheet reaches the sphere it is flat and every particle moves
    // identically, so no spring is stretched and no spring force exists. What
    // remains is gravity and the global air drag, integrated by semi-implicit
    // Euler, which has an exact closed form:
    //
    //     v(n+1) = v(n) * (1 - (drag/mass) * dt) - g * dt
    //     v(n)   = terminal * (1 - decay^n)
    //     y(n)   = y(0) + dt * sum of v(1..n)
    //
    // Agreement pins gravity, the drag coefficient, the mass, the time step and
    // the integration scheme at once. Any of them changing breaks this.
    let Some(gpu) =
        harness::gpu_or_skip("a_relaxed_sheet_falls_exactly_as_the_equations_say_it_should")
    else {
        return;
    };
    let config = harness::small_config();
    let physics = config.physics();

    // f64 throughout, so the tolerance measures the simulation rather than the
    // arithmetic checking it.
    let step = f64::from(gpu_cloth_simulation::simulation::FIXED_TIME_STEP_SECONDS);
    let decay = 1.0 - (f64::from(physics.damping) / f64::from(physics.mass)) * step;
    let terminal = -f64::from(GRAVITY) * step / (1.0 - decay);

    // 500 steps: the sheet is still above the sphere, so nothing has touched.
    for count in [1u32, 10, 100, 500] {
        let particles = harness::simulate(&gpu, &config, count);
        let corner = particles[0];

        let n = f64::from(count);
        let fallen = decay.powf(n);
        let speed = terminal * (1.0 - fallen);
        let height = f64::from(RELEASE_HEIGHT)
            + step * terminal * (n - decay * (1.0 - fallen) / (1.0 - decay));

        // Three orders of magnitude above the f32 round-off actually observed,
        // and far below any change of physics.
        let tolerance = 1e-5;
        let speed_error = (f64::from(corner.speed[1]) - speed).abs() / speed.abs();
        let height_error = (f64::from(corner.position[1]) - height).abs() / height.abs();
        assert!(
            speed_error < tolerance,
            "after {count} steps the corner falls at {} where the equations say \
             {speed}, a relative error of {speed_error}",
            corner.speed[1]
        );
        assert!(
            height_error < tolerance,
            "after {count} steps the corner is at {} where the equations say \
             {height}, a relative error of {height_error}",
            corner.position[1]
        );
    }
}

#[test]
fn nothing_falls_through_the_ground() {
    let Some(gpu) = harness::gpu_or_skip("nothing_falls_through_the_ground") else {
        return;
    };
    let particles = harness::simulate(&gpu, &harness::small_config(), 3_000);
    for (index, particle) in particles.iter().enumerate() {
        assert!(
            particle.position[1] >= GROUND - 1e-4,
            "particle {index} is below the ground at y = {}",
            particle.position[1]
        );
    }
}

#[test]
fn nothing_ends_up_inside_the_obstacle_sphere() {
    // The sphere collision projects a particle back onto the surface. If it
    // does not, the cloth sinks through the obstacle it is draping over.
    let Some(gpu) = harness::gpu_or_skip("nothing_ends_up_inside_the_obstacle_sphere") else {
        return;
    };
    let config = harness::small_config();
    let particles = harness::simulate(&gpu, &config, 3_000);
    // One step of integration can move a particle slightly inward after the
    // projection, so the tolerance is one step at a generous speed.
    let tolerance = 0.02;
    for (index, particle) in particles.iter().enumerate() {
        assert!(
            harness::radius(particle) >= config.sphere_radius - tolerance,
            "particle {index} is {} from the origin, inside the {} sphere",
            harness::radius(particle),
            config.sphere_radius
        );
    }
}

#[test]
fn no_particle_is_flung_above_where_it_started() {
    // A mass-spring sheet that gains energy launches particles upward. Nothing
    // in this scene pushes up except the ground and the sphere, and neither can
    // return a particle above its release height.
    let Some(gpu) = harness::gpu_or_skip("no_particle_is_flung_above_where_it_started") else {
        return;
    };
    let particles = harness::simulate(&gpu, &harness::small_config(), 3_000);
    for (index, particle) in particles.iter().enumerate() {
        assert!(
            particle.position[1] <= RELEASE_HEIGHT + 1e-3,
            "particle {index} rose to y = {}, above the {RELEASE_HEIGHT} it was released from",
            particle.position[1]
        );
    }
}

#[test]
fn the_sheet_stays_bounded_in_the_horizontal_plane() {
    // The cloth drapes; it does not expand. A sheet whose springs are pumping
    // energy spreads sideways without limit long before it produces a NaN.
    let Some(gpu) = harness::gpu_or_skip("the_sheet_stays_bounded_in_the_horizontal_plane") else {
        return;
    };
    let config = harness::small_config();
    let side = ClothSimulation::new(&gpu.device, &config).grid_size();
    // Half the initial width, with room for the drape around a 0.4 sphere.
    let bound = (side as f32 * config.spacing) / 2.0 + config.sphere_radius + 0.5;

    let particles = harness::simulate(&gpu, &config, 3_000);
    for (index, particle) in particles.iter().enumerate() {
        let horizontal = (particle.position[0].powi(2) + particle.position[2].powi(2)).sqrt();
        assert!(
            horizontal <= bound,
            "particle {index} is {horizontal} from the vertical axis, past the {bound} bound"
        );
    }
}

#[test]
fn the_simulation_is_deterministic() {
    // Same configuration, same number of steps, same result: without this,
    // every other assertion here is only true of the run that produced it.
    let Some(gpu) = harness::gpu_or_skip("the_simulation_is_deterministic") else {
        return;
    };
    let config = harness::small_config();
    let first = harness::simulate(&gpu, &config, 300);
    let second = harness::simulate(&gpu, &config, 300);
    assert_eq!(first.len(), second.len());
    for (index, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert_eq!(a, b, "particle {index} differs between two identical runs");
    }
}

#[test]
fn every_particle_is_stepped() {
    // The dispatch rounds the workgroup count up and the shader discards the
    // surplus invocations. If either half of that were wrong, the tail of the
    // buffer would simply never move, which no visual check would catch on a
    // sheet of thousands of particles.
    let Some(gpu) = harness::gpu_or_skip("every_particle_is_stepped") else {
        return;
    };
    let config = harness::small_config();
    let mut simulation = ClothSimulation::new(&gpu.device, &config);
    assert!(
        simulation.particle_count() % WORKGROUP_SIZE != 0,
        "this test is only worth running on a grid with a partial workgroup"
    );
    let before = simulation.read_particles(&gpu.device, &gpu.queue);
    for _ in 0..200 {
        simulation.step(&gpu.device, &gpu.queue);
    }
    let after = simulation.read_particles(&gpu.device, &gpu.queue);

    let untouched: Vec<usize> = before
        .iter()
        .zip(after.iter())
        .enumerate()
        .filter(|(_, (b, a))| b.position == a.position)
        .map(|(index, _)| index)
        .collect();
    assert!(
        untouched.is_empty(),
        "{} particles never moved, first at index {:?}",
        untouched.len(),
        untouched.first()
    );
}

#[test]
fn the_grid_size_asked_for_is_the_grid_size_simulated() {
    // The slider offers 64 to 512 in steps of 64, and the grid side was rounded
    // down to a multiple of the workgroup size so the dispatch divided evenly.
    // Every request below 256 therefore became 256: asking for a 64 x 64 sheet
    // built a 256 x 256 one, sixteen times the particles, and the number on the
    // slider was not the number being simulated.
    let Some(gpu) = harness::gpu_or_skip("the_grid_size_asked_for_is_the_grid_size_simulated")
    else {
        return;
    };
    for requested in [64u32, 128, 192, 256, 320, 512] {
        let config = ClothConfig {
            grid_size: requested,
            ..harness::small_config()
        };
        let simulation = ClothSimulation::new(&gpu.device, &config);
        assert_eq!(
            simulation.grid_size(),
            requested,
            "asked for a {requested} x {requested} sheet"
        );
        assert_eq!(
            simulation.read_particles(&gpu.device, &gpu.queue).len(),
            (requested * requested) as usize,
            "particle count for a {requested} x {requested} sheet"
        );
    }
}

#[test]
fn a_grid_that_does_not_fill_its_workgroups_still_simulates() {
    // 100 x 100 is 10 000 particles, which is 39.06 workgroups. The dispatch
    // rounds up, so the last workgroup runs invocations with no particle behind
    // them and the shader has to discard them rather than read past the buffer.
    let Some(gpu) =
        harness::gpu_or_skip("a_grid_that_does_not_fill_its_workgroups_still_simulates")
    else {
        return;
    };
    let config = ClothConfig {
        grid_size: 100,
        ..harness::small_config()
    };
    let before = ClothSimulation::new(&gpu.device, &config).read_particles(&gpu.device, &gpu.queue);
    let after = harness::simulate(&gpu, &config, 500);

    assert_eq!(after.len(), 10_000);
    for (index, particle) in after.iter().enumerate() {
        assert!(
            particle
                .position
                .iter()
                .chain(particle.speed.iter())
                .all(|v| v.is_finite()),
            "particle {index} is not finite: {particle:?}"
        );
    }
    assert!(
        harness::mean_height(&after) < harness::mean_height(&before),
        "the sheet did not fall, so the surplus invocations may have eaten the real ones"
    );
}

#[test]
fn the_friction_coefficient_reaches_the_shader() {
    // repaired: PhysicsParams carried a `friction` field, uploaded on every
    // build, that the shader never read. The collision code used a local
    // `let cf = 0.9` instead, under a comment saying so. The parameter was
    // therefore inert, and nothing would have noticed if it had been removed.
    let Some(gpu) = harness::gpu_or_skip("the_friction_coefficient_reaches_the_shader") else {
        return;
    };
    let slippery = ClothConfig {
        friction: 0.0,
        ..harness::small_config()
    };
    let grippy = ClothConfig {
        friction: 1.0,
        ..harness::small_config()
    };

    // Long enough for the sheet to reach the sphere, which is the only place
    // friction is applied.
    let with_none = harness::simulate(&gpu, &slippery, 2_000);
    let with_lots = harness::simulate(&gpu, &grippy, 2_000);

    assert!(
        with_none
            .iter()
            .zip(&with_lots)
            .any(|(a, b)| a.position != b.position),
        "the sheet settles identically with no friction and with full friction: \
         the coefficient is not reaching the shader"
    );
}

// The two tests below need the heaviest configuration the interface offers,
// 262,144 particles run for thousands of steps: that is the only place the sheet
// stretches far enough to exercise the cap at all. On a real GPU it is seconds.
// On the software rasteriser a CI runner has, it would be the whole build. So
// they are ignored by default, run nightly, and run on demand with:
//
//     cargo test --test physics -- --ignored

/// The tightest spacing and the largest grid the UI offers, which together load
/// the springs hardest and are where every stability problem showed up first.
fn heaviest_supported_config() -> ClothConfig {
    ClothConfig {
        grid_size: 512,
        spacing: 0.002,
        ..harness::small_config()
    }
}

#[test]
#[ignore = "262,144 particles for 12,000 steps: see the note above"]
fn a_settled_sheet_never_reaches_the_stretch_cap() {
    // The cap is a safety net, not a physics knob. A sheet hanging at rest must
    // stay under it, or the constraint fires on every step forever: it is then
    // fighting statics rather than catching a divergence, and the correction it
    // feeds back into the velocity accumulates until the sheet goes non-finite.
    // Measured with the constraint disabled, a resting sheet stretches up to
    // 1.365x over the whole supported range, against a cap of 1.5.
    let Some(gpu) = harness::gpu_or_skip("a_settled_sheet_never_reaches_the_stretch_cap") else {
        return;
    };
    let config = heaviest_supported_config();
    let mut simulation = ClothSimulation::new(&gpu.device, &config);
    let side = simulation.grid_size() as usize;
    for _ in 0..12_000 {
        simulation.step(&gpu.device, &gpu.queue);
    }
    let settled = simulation.read_particles(&gpu.device, &gpu.queue);

    let worst = harness::worst_stretch(&settled, side, config.spacing);
    assert!(
        worst < MAX_SPRING_STRETCH,
        "a settled sheet stretches {worst}x, at or past the {MAX_SPRING_STRETCH}x cap: \
         the constraint has nothing left to catch and will fight gravity forever"
    );
}

#[test]
#[ignore = "262,144 particles for 10,000 steps: see the note above"]
fn the_stretch_cap_clips_the_transient() {
    // The mechanism tested in the conditions it exists for. As the falling sheet
    // snaps taut on the sphere it reaches 3.04x its rest length unconstrained,
    // which is not cloth. The constraint has to bring that down, and this
    // compares the two runs rather than trusting the cap to be respected.
    let Some(gpu) = harness::gpu_or_skip("the_stretch_cap_clips_the_transient") else {
        return;
    };
    let config = heaviest_supported_config();

    let peak = |iterations: u32| {
        let config = ClothConfig {
            constraint_iterations: iterations,
            ..config
        };
        let mut simulation = ClothSimulation::new(&gpu.device, &config);
        let side = simulation.grid_size() as usize;
        let mut peak = 0.0f32;
        for _ in 0..100 {
            for _ in 0..50 {
                simulation.step(&gpu.device, &gpu.queue);
            }
            let state = simulation.read_particles(&gpu.device, &gpu.queue);
            peak = harness::larger(peak, harness::worst_stretch(&state, side, config.spacing));
        }
        peak
    };

    let unconstrained = peak(0);
    let constrained = peak(CONSTRAINT_ITERATIONS);

    assert!(
        unconstrained > 2.0,
        "this configuration is supposed to over-stretch without the constraint, \
         but it only reached {unconstrained}x: the test no longer exercises the cap"
    );
    assert!(
        constrained < unconstrained * 0.75,
        "the constraint brought the peak from {unconstrained}x only to {constrained}x: \
         it is not clipping the transient it exists for"
    );
}

/// Downward acceleration in `compute.wgsl`, in scene units per second squared.
const GRAVITY: f32 = 0.3;

/// The fastest anything in this scene can legitimately travel.
///
/// Nothing pushes the cloth: it is released at rest and falls, so a free fall
/// over the whole drop, from the release height to the ground, is the scale of
/// every legitimate speed in the scene. Springs snapping taut overshoot it: the
/// worst measured across the supported range is 2.5x. The factor of ten below
/// leaves room for that while staying four orders of magnitude under a runaway,
/// which passed 5e4 within a few steps of starting.
fn free_fall_speed_limit() -> f32 {
    10.0 * (2.0 * GRAVITY * (RELEASE_HEIGHT - GROUND)).sqrt()
}

#[test]
fn no_particle_outruns_a_free_fall_from_the_release_height() {
    // repaired: the positional constraint moved a particle back inside the
    // stretch cap but left its velocity untouched, so the projected position and
    // the stored velocity disagreed. The spring stayed pinned at the cap, the
    // same correction was applied again on every step, and the speed grew without
    // bound until the sheet went non-finite around step 1330. Checking only the
    // end state hid it: the cap itself still read exactly 1.1000 throughout.
    let Some(gpu) = harness::gpu_or_skip("no_particle_outruns_a_free_fall_from_the_release_height")
    else {
        return;
    };
    let config = harness::small_config();
    let limit = free_fall_speed_limit();
    let mut simulation = ClothSimulation::new(&gpu.device, &config);

    for block in 1..=28 {
        for _ in 0..250 {
            simulation.step(&gpu.device, &gpu.queue);
        }
        let particles = simulation.read_particles(&gpu.device, &gpu.queue);
        let fastest = harness::top_speed(&particles);
        assert!(
            fastest <= limit,
            "after {} steps the fastest particle reached {fastest}, past the {limit} \
             that ten free falls from the release height allow",
            block * 250
        );
    }
}

#[test]
fn the_cloth_settles_instead_of_oscillating_forever() {
    // Damping is supposed to bleed energy out. If it does not, the sheet keeps
    // moving indefinitely and never drapes.
    let Some(gpu) = harness::gpu_or_skip("the_cloth_settles_instead_of_oscillating_forever") else {
        return;
    };
    let config = harness::small_config();
    let mut simulation = ClothSimulation::new(&gpu.device, &config);

    for _ in 0..1_500 {
        simulation.step(&gpu.device, &gpu.queue);
    }
    let early = harness::kinetic_energy(&simulation.read_particles(&gpu.device, &gpu.queue));
    for _ in 0..6_000 {
        simulation.step(&gpu.device, &gpu.queue);
    }
    let late = harness::kinetic_energy(&simulation.read_particles(&gpu.device, &gpu.queue));

    assert!(
        late < early,
        "total kinetic energy went from {early} to {late}: the cloth is not settling"
    );
}
