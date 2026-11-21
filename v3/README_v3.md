# v3: Closed-loop simulation grounded in the published control law

This version is built directly from the actual published paper (Wang, Yu,
Matthews, Alexander, Kriegman, Rubenstein — "Computational Design of a
Low-Visibility UAV Using a Human-Aligned Perceptual Metric," arXiv:2605.11296),
not just from email correspondence. Two things from the paper changed the
simulation meaningfully:

1. **The real control law** (Section III-B): "The motor executes two power
   commands per rotation: the average of commands regulates vertical
   thrust, while the difference of commands and the switching time
   regulate the average torque that tilts the Phantom Twist." The
   controller decides *when* in the rotation to switch power levels based
   on its own phase estimate — not the true phase.
2. **A stated failure threshold**, not an estimated one: "a timing error
   exceeding 5ms can cause loss of control."

## Why this is a different (and more honest) simulation than v2

v2 treated the gap between commanded and true dynamics as an **independent
external disturbance** — plausible, but not actually how the real system
works. This paper's control law reveals the real mechanism: since
switching is timed against the *estimated* phase, a wrong phase estimate
causes the real thrust vector to be **physically misdirected** by
approximately the phase error itself. The estimator's own error becomes a
genuine disturbance to the real trajectory — a closed loop, not two
independent processes.

v3 simulates this directly: true acceleration is computed by rotating the
intended thrust direction by the EKF's own current phase error, while the
EKF's prediction step uses what the controller *believes* it's doing
(since a controller cannot know its own estimation error). This also
enabled using the paper's own 5ms timing-error criterion as a pass/fail
metric, instead of an estimated "a few mm" accuracy bar.

## Results (20 Monte Carlo seeds/level, 15s flight, 20 rev/s nominal spin)

| Spin disturbance | Final error, no correction | Final error, EKF | Time over 5ms threshold, no correction | Time over 5ms threshold, EKF |
|---|---|---|---|---|
| 0.05 | 4.3 ± 3.7 m | 0.34 ± 0.21 m | 45% | **0%** |
| 0.15 | 10.4 ± 9.9 m | 0.32 ± 0.19 m | 78% | **0%** |
| 0.30 | 13.8 ± 10.8 m | 0.36 ± 0.19 m | 89% | **0%** |
| 0.50 | 14.5 ± 11.2 m | 0.44 ± 0.19 m | 93% | **0%** |

**The headline finding:** across every disturbance level tested, the EKF
kept the vehicle's phase estimate within the paper's own stated stability
margin 100% of the time. Without correction, the vehicle would spend
45–93% of the flight in a state the paper's own authors say can cause loss
of control — and phase error grows essentially without bound (hundreds of
degrees by the end of some runs), since nothing is correcting it.

This is a stronger and more defensible claim than v2's — it's not "the
EKF is more accurate," it's "the EKF keeps the vehicle inside a
real, independently-stated flight-safety margin that the open-loop
approach does not."

## Honest limitations

- Still 2D, not 3D.
- The rotation of realized thrust by exactly the phase error is a
  simplification of the true two-level switching mechanism's effect —
  the real relationship likely involves duty-cycle and switching-phase
  details not modeled here (would need the full Fourier/averaging
  analysis of the square-wave thrust profile to get exactly right).
- The accuracy bar Jingxian originally cited ("several mm") still hasn't
  been directly reconciled with the paper's own reported 3cm RMS / 10cm
  max hover error using their actual external tracking system — worth
  asking her directly whether that few-mm figure is about total hover
  error or specifically about how much additional error an estimator is
  allowed to contribute on top of what's already in the real system.
- Vision noise is still white noise, not the colored-noise model
  identified as a to-do item after her second round of feedback — that
  remains a real next step, now with a stronger baseline to add it to.

## Files

- `phantom_twist_v3.py` — closed-loop simulation and EKF.
- `run_v3_monte_carlo.py` / `v3_results.json` — 20-seed validation across
  disturbance levels.
- `plot_v3_results.py` / `v3_results.png` — the two plots above.
