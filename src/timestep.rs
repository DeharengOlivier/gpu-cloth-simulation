//! Turning elapsed wall-clock time into a whole number of fixed physics steps.
//!
//! Pure arithmetic on its own state: no device, no queue, no frame. That is what
//! makes the two things it has to get right testable without a GPU.

/// Physics steps a single frame may run before the rest of its time is dropped.
///
/// A frame that has to catch up on more than this is a frame that already took
/// too long. Running every step it asks for makes the next frame later still,
/// which asks for more steps again: the simulation stops responding and the
/// window stops repainting. Dropping the surplus instead means the cloth runs
/// in slow motion during a hitch, which the user can see and recover from.
///
/// At the default step of 1.6 ms this is 64 ms of catch-up, about four frames
/// at 60 Hz.
pub const MAX_STEPS_PER_FRAME: u32 = 40;

/// Accumulates frame time and hands out whole physics steps.
#[derive(Debug, Default)]
pub struct FixedTimestep {
    /// Simulated time owed but not yet run, always less than one step once
    /// [`Self::steps_for`] has returned.
    unspent: f32,
}

impl FixedTimestep {
    /// How many steps to run for a frame that took `delta_time` seconds.
    ///
    /// The remainder is kept, so simulated time tracks elapsed time instead of
    /// losing a fraction of a step every frame. A frame asking for more than
    /// [`MAX_STEPS_PER_FRAME`] gets the cap and the rest of its time is dropped
    /// rather than owed, which is what stops a hitch from repeating itself.
    ///
    /// A non-positive `delta_time` contributes nothing: a clock that jumps
    /// backwards must not put the accumulator into debt.
    ///
    /// Complexity: O(1).
    pub fn steps_for(&mut self, delta_time: f32, step: f32) -> u32 {
        debug_assert!(step > 0.0, "the physics step must be positive");
        if delta_time > 0.0 {
            self.unspent += delta_time;
        }

        let owed = (self.unspent / step) as u32;
        let steps = owed.min(MAX_STEPS_PER_FRAME);
        if steps < owed {
            self.unspent = 0.0;
        } else {
            self.unspent -= steps as f32 * step;
        }
        steps
    }

    /// Simulated time owed but not yet run.
    pub fn unspent(&self) -> f32 {
        self.unspent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: f32 = 0.0016;

    #[test]
    fn a_frame_too_short_for_one_step_runs_nothing() {
        let mut clock = FixedTimestep::default();
        assert_eq!(clock.steps_for(STEP / 2.0, STEP), 0);
    }

    #[test]
    fn time_left_over_by_one_frame_is_spent_by_the_next() {
        // The accumulator used to be a local, so every frame threw its remainder
        // away. At 60 Hz that is 0.67 ms of the 16.7 ms dropped every frame, and
        // the cloth ran 4% slow while the comment above the loop claimed the
        // rate was independent of the frame rate.
        let mut clock = FixedTimestep::default();
        let frame = STEP * 0.6;
        assert_eq!(clock.steps_for(frame, STEP), 0);
        assert_eq!(clock.steps_for(frame, STEP), 1);
    }

    #[test]
    fn simulated_time_keeps_up_with_elapsed_time_over_many_frames() {
        let mut clock = FixedTimestep::default();
        let frame = 1.0 / 60.0;
        let frames = 600;
        let steps: u32 = (0..frames).map(|_| clock.steps_for(frame, STEP)).sum();
        let owed = (frames as f32 * frame / STEP) as u32;
        assert!(
            steps.abs_diff(owed) <= 1,
            "ran {steps} steps for {owed} steps' worth of frames"
        );
    }

    #[test]
    fn one_long_frame_cannot_run_an_unbounded_number_of_steps() {
        // The loop was `while accumulated >= step`, so a frame delayed by a
        // breakpoint, a dragged window or a sleeping laptop asked for every step
        // it had missed at once, which delayed the next frame further.
        let mut clock = FixedTimestep::default();
        assert_eq!(clock.steps_for(10.0, STEP), MAX_STEPS_PER_FRAME);
    }

    #[test]
    fn the_time_dropped_by_a_hitch_is_not_owed_forever() {
        // Capping the steps but keeping the debt would run the cap again on
        // every following frame, which is the same spiral one frame later.
        let mut clock = FixedTimestep::default();
        clock.steps_for(10.0, STEP);
        assert!(
            clock.unspent() < STEP,
            "still owes {} seconds after the hitch",
            clock.unspent()
        );
        assert_eq!(clock.steps_for(STEP * 2.0, STEP), 2);
    }

    #[test]
    fn a_frame_that_did_not_advance_runs_nothing() {
        let mut clock = FixedTimestep::default();
        assert_eq!(clock.steps_for(0.0, STEP), 0);
    }

    #[test]
    fn time_never_runs_backwards() {
        // wgpu_bootstrap hands the frame time straight through, so a clock that
        // jumps must not put the accumulator into debt it can never repay.
        let mut clock = FixedTimestep::default();
        assert_eq!(clock.steps_for(-1.0, STEP), 0);
        assert_eq!(clock.steps_for(STEP, STEP), 1);
    }
}
