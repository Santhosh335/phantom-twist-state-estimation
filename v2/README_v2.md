# v2: Onboard state estimation without a gyro

This revises the original EKF demo based on direct feedback from Jingxian
Wang (co-first author, Phantom Twist). Her correction of my original
assumption is the reason this version exists, and I want that to be
visible rather than quietly folded in.

## What changed, and why

The v1 demo assumed a working onboard rate gyro with a small constant
bias. Jingxian corrected that assumption directly: at 15-25 rev/s
(roughly 5,400-9,000 deg/s), consumer MEMS gyros saturate outright - they
don't just get noisy, they stop reporting usable data. The only gyro she
found that can survive this rotation rate is ~10g and $200+ (used on
their MP2 prototype), which is not viable on a gram-scale aircraft. In
her words: **"we don't even have IMU on board."**

She also confirmed their own working localization approach - relative
position measurements between units, refined with a KF (used on MP3,
improved on the unreleased MP4) - requires an external reference (another
unit or reference point to measure against). That means it does not solve
the case this project is specifically about: a single, solo drone with
nothing external to reference. That case is still open, even inside the
team that built the hardware.

## The revised approach

Drop the gyro (and the IMU) entirely. State estimation now works from:

1. **A dynamics model driven by known commanded inputs** - the spin rate
   and thrust the flight controller already intends to command are
   exactly known (no sensor needed to "measure" your own commands). This
   replaces gyro+accelerometer integration as the prediction step.
2. **Vision (optical flow) as the only aiding sensor** - the same role it
   played in v1, correcting for whatever the real airframe does
   differently from what was commanded (wind gusts, motor response lag,
   imperfect thrust-pulse timing).

State: `[px, py, vx, vy, theta, spin_bias]`. `spin_bias` now tracks the
gap between *commanded* and *true* spin rate (a real physical disturbance),
not a sensor error - there is no sensor to have an error.

## A real bug, fixed honestly

The first version of this filter made things *worse* than doing nothing:
the EKF's process noise was tuned as if the model could be arbitrarily
wrong, which made it over-trust noisy vision measurements even when the
model was already accurate. Fixing the tuning closed most of the gap -
but a second, more interesting problem remained: when the real spin-rate
disturbance was large, a single slowly-adapting bias state couldn't keep
up with it. Once the phase estimate fell behind by even a fraction of a
rotation, every subsequent vision correction got applied through the
wrong rotation and actively corrupted the estimate rather than fixing it.

The fix was to scale the bias state's process noise to the expected
disturbance level - a real, practical tuning requirement worth stating
plainly rather than hiding: **this filter needs to know roughly how
gusty its environment will be before it can be trusted.** That's a
limitation, not just a result.

## Results (30 seeds/scenario, 20 rev/s spin, 8s flight)

| Scenario | Model only (final error) | EKF (final error) | EKF wins |
|---|---|---|---|
| Calm (minimal disturbance) | 0.009 ± 0.005 m | 0.013 ± 0.007 m | 13/30 |
| Gusty (realistic outdoor) | 0.052 ± 0.031 m | 0.017 ± 0.010 m | 29/30 |

This is a more honest result than a blanket "EKF always wins" claim, and
I think that's the right story to tell: in calm conditions the open-loop
model is already good enough, and fusing noisy vision can even hurt
slightly. The EKF's value shows up specifically once real-world
disturbance exceeds what the model alone can track - see
`sensitivity_v2.png` for where that crossover happens (around a
disturbance level of ~0.15-0.2 in this parameterization).

## Honest limitations (unchanged from v1, still true)

- 2D simulation, not 3D, not hardware.
- Vision is modeled as clean rotated-velocity measurement with additive
  noise - a real rolling-shutter camera on a 20Hz-spinning body is a
  vision problem of its own.
- The commanded world-frame thrust is treated as directly achievable,
  abstracting away the phase-timed thrust-pulse controller needed to
  realize it - a real implementation would need that controller to also
  depend on the phase estimate this filter produces, which creates a
  closed loop this simulation doesn't yet model.
- `spin_bias` process noise must be hand-tuned to the expected
  disturbance level; this filter doesn't yet adapt that online.

## Files

- `phantom_twist_ekf_v2.py` - ground truth, flow sensor model, open-loop
  baseline, and the model-based EKF.
- `run_single_demo_v2.py` / `single_run_v2.png`
- `run_monte_carlo_v2.py` / `monte_carlo_v2.png` - calm vs. gusty, 30 seeds each
- `run_sensitivity_v2.py` / `sensitivity_v2.png` - disturbance-level sweep
