> **Update (Oct 2026):** This was revised based on direct feedback from Jingxian Wang (co-first author, Phantom Twist) — the original design assumed a working onboard gyro, which doesn't survive the drone's spin rate. See v2/README_v2.md for the corrected approach, the bug I found and fixed, and the updated results.
# Onboard state estimation for a spinning-body UAV (Phantom Twist)

A simulation study of self-localization for a Phantom-Twist-style drone
(Northwestern, RSS 2026) without an external motion-capture rig.

## Motivation

Phantom Twist achieves near-invisibility by spinning its entire airframe
at 15-25 rev/s in the direction opposite its single propeller, exploiting
persistence-of-vision. The published prototype currently needs an
external motion-capture system to know its own position in space, since
it has no onboard navigation. This repository is a first attempt at the
onboard alternative.

The core difficulty isn't localization in the abstract - it's that every
candidate onboard sensor (IMU, camera) is rigidly bolted to a body that
is itself spinning faster than most estimators are designed to handle.
A gyro bias of even a few hundredths of a rad/s, integrated at ~130
rad/s, misaligns the estimated spin phase within seconds - and that
phase error then corrupts every other sensor reading rotated through it.

## Approach

Two estimators, compared head-to-head on identical simulated sensor data:

1. **Dead reckoning (baseline)** - integrate gyro + body-frame
   accelerometer only. No correction, no bias estimate. This is
   approximately what you get from a naive IMU-only implementation.

2. **EKF (this work)** - a 6-state extended Kalman filter
   `[px, py, vx, vy, theta, gyro_bias]` that:
   - uses the gyro + accelerometer as prediction inputs (strapdown-INS
     style), explicitly estimating gyro bias as a state rather than
     assuming it away,
   - corrects using a body-frame optical-flow velocity measurement -
     the only aiding sensor assumed available once mocap is removed.

Full derivation (rotation Jacobians for the prediction and measurement
models) is in `phantom_twist_ekf.py`.

## Results (30 Monte Carlo seeds, 20 rev/s spin, 0.05 rad/s gyro bias, 8s flight)

| Estimator       | Position RMSE (m) | Final position error (m) |
|-----------------|--------------------|---------------------------|
| Dead reckoning  | 0.248 ± 0.010      | 0.671 ± 0.023             |
| EKF             | 0.181 ± 0.032      | 0.353 ± 0.072             |

- EKF reduces mean final position error by **47%** and wins on
  **30/30** individual seeds (not just on average).
- The gyro bias state converges to within ~3% of the true injected bias.
- Dead-reckoning error grows unbounded (bias-driven, roughly quadratic);
  EKF error is bounded by the optical-flow noise floor rather than by
  drift - see `single_run.png` for the crossover behavior.
- The EKF stays within ~3% of its nominal accuracy even with **80%** of
  optical-flow measurements dropped (simulating motion blur / occlusion
  / low-texture ground) - see `robustness.png`. This matters because a
  real camera on a body spinning at 20 rev/s *will* lose lock regularly.

## Honest limitations (next steps, not hidden)

- This is a 2D planar simplification. Phantom Twist operates with a real
  3D attitude and a passively-stabilized spin axis; extending to a full
  3D quaternion-based filter is the natural next step.
- Optical flow here is simulated as a clean rotated-velocity measurement
  with additive noise. A real implementation needs to handle flow
  computed from a rolling-shutter camera on a body rotating at 20 Hz,
  which is a nontrivial vision problem on its own.
- Accelerometer bias is not yet a filter state (only gyro bias is) -
  worth adding given how much gyro bias alone mattered here.
- No hardware validation - this is a simulation study intended to
  demonstrate the estimator is sound before committing lab time to it.

## Files

- `phantom_twist_ekf.py` - ground truth simulation, sensor models, dead
  reckoning baseline, and the EKF implementation.
- `run_single_demo.py` / `single_run.png` - one representative run with
  trajectory and error-over-time plots.
- `run_monte_carlo.py` / `monte_carlo.png` - 30-seed statistical
  validation.
- `run_robustness.py` / `robustness.png` - sensitivity to optical-flow
  dropout rate.

## Reproducing

```
pip install numpy matplotlib
python run_single_demo.py
python run_monte_carlo.py
python run_robustness.py
```
