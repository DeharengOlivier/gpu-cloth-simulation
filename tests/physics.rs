//! What the compute shader is supposed to guarantee, asserted rather than watched.
//!
//! The README used to say the physics was "validated visually by running the
//! simulation", and that a headless harness "does not exist yet". It does now:
//! each test below runs the real `compute.wgsl` on a real device with no window,
//! reads the particle buffer back, and checks an invariant the simulation claims.

mod harness;

use gpu_cloth_simulation::simulation::{ClothSimulation, WORKGROUP_SIZE};

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
    // The dispatch covers particle_count / WORKGROUP_SIZE workgroups. If the
    // arithmetic were ever wrong, the tail of the buffer would simply never
    // move, which no visual check would catch on a 65,536-particle sheet.
    let Some(gpu) = harness::gpu_or_skip("every_particle_is_stepped") else {
        return;
    };
    let config = harness::small_config();
    let mut simulation = ClothSimulation::new(&gpu.device, &config);
    assert_eq!(
        simulation.particle_count() % WORKGROUP_SIZE,
        0,
        "a partial workgroup would leave particles unstepped"
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
