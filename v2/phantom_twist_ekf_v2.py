"""
Onboard state estimation for a Phantom-Twist-style spinning UAV - v2.

Why this version exists
------------------------
v1 assumed a working onboard rate gyro with a small constant bias. Direct
correspondence with the Phantom Twist team (Jingxian Wang, co-first author)
corrected that assumption: at 15-25 rev/s (up to ~9000 deg/s), consumer
MEMS gyros saturate outright. The only gyro that can survive this rotation
rate is ~10g and $200+ (used on their MP2 prototype) - effectively
unusable on a gram-scale aircraft. "We don't even have IMU on board."

She also confirmed their own localization approach (relative-position
measurements between units, filtered with a KF, as used on MP3/MP4)
requires an external reference - meaning it does not solve the standalone,
no-external-reference case this project is specifically about. That case
remains open.

v2 therefore drops the gyro (and, by extension, the IMU) entirely and
asks a different question: if you don't have a gyro, can you still track
a spinning body's state using only (a) the dynamics model implied by the
motor commands you already know you sent, and (b) a single aiding sensor
- vision (optical flow) - to correct for whatever the real airframe does
differently from what you commanded?

State: x = [px, py, vx, vy, theta, spin_bias]  (6,)
  spin_bias tracks the *actual* mismatch between commanded and true spin
  rate (motor response lag, aerodynamic disturbance) - not a sensor bias,
  since there is no sensor to have a bias.

Prediction uses KNOWN COMMANDED inputs (omega_cmd, a_cmd) - not noisy
measurements - because these are set by the flight controller itself, so
they are exactly known. The uncertainty being tracked is "how well does
reality follow the command," which only vision can resolve.

Correction uses body-frame optical flow (same measurement model as v1).
"""

import numpy as np


def R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------
# 1. Ground truth trajectory (true spin/accel differ from commanded)
# ----------------------------------------------------------------------

def simulate_ground_truth(dt, T, spin_rate_rps=20.0, spin_up_time=0.5,
                           accel_amp=0.35, accel_omega=0.5,
                           spin_mismatch=0.05, accel_mismatch_frac=0.03,
                           seed=0):
    """
    Commanded profile: what the flight controller intends (known exactly).
    True profile: what the airframe actually does - offset from command by
    a slowly-varying disturbance (motor response lag, aero drag, timing
    jitter in the thrust-pulse controller). This gap is exactly what the
    estimator has to discover using vision, since there is no gyro to
    measure it directly.
    """
    rng = np.random.default_rng(seed)
    n = int(T / dt) + 1
    t = np.linspace(0, T, n)

    omega_target = 2 * np.pi * spin_rate_rps
    omega_cmd = np.where(t < spin_up_time, omega_target * (t / spin_up_time), omega_target)

    # true spin rate = commanded + a slow random-walk disturbance
    spin_disturbance = np.cumsum(rng.normal(0, spin_mismatch * np.sqrt(dt), n))
    omega_true = omega_cmd + spin_disturbance

    theta_true = np.concatenate(([0.0], np.cumsum((omega_true[:-1] + omega_true[1:]) / 2 * dt)))
    theta_cmd = np.concatenate(([0.0], np.cumsum((omega_cmd[:-1] + omega_cmd[1:]) / 2 * dt)))

    a_cmd = np.stack([
        accel_amp * np.cos(accel_omega * t),
        accel_amp * np.sin(accel_omega * t)
    ], axis=1)
    # true thrust differs from commanded by a small multiplicative + noise term
    a_true = a_cmd * (1 + accel_mismatch_frac * rng.normal(0, 1, (n, 1))) \
        + rng.normal(0, 0.01, (n, 2))

    v_world = np.zeros((n, 2))
    p_world = np.zeros((n, 2))
    for k in range(1, n):
        v_world[k] = v_world[k - 1] + a_true[k - 1] * dt
        p_world[k] = p_world[k - 1] + v_world[k - 1] * dt + 0.5 * a_true[k - 1] * dt ** 2

    return {
        't': t, 'dt': dt, 'n': n,
        'theta': theta_true, 'omega': omega_true,
        'theta_cmd': theta_cmd, 'omega_cmd': omega_cmd,
        'p': p_world, 'v': v_world, 'a': a_true, 'a_cmd': a_cmd,
    }


# ----------------------------------------------------------------------
# 2. Vision sensor (the only sensor assumed present)
# ----------------------------------------------------------------------

def generate_flow_sensor(truth, flow_noise_std=0.06, flow_dropout_prob=0.0, seed=0):
    rng = np.random.default_rng(seed + 1000)
    n = truth['n']
    flow_body_true = np.einsum('nij,nj->ni',
                                np.stack([R(-th) for th in truth['theta']]),
                                truth['v'])
    flow_meas = flow_body_true + rng.normal(0, flow_noise_std, (n, 2))
    flow_valid = rng.random(n) > flow_dropout_prob
    return {'flow': flow_meas, 'flow_valid': flow_valid, 'flow_noise_std': flow_noise_std}


# ----------------------------------------------------------------------
# 3. Pure model-based dead reckoning (baseline - no vision correction)
# ----------------------------------------------------------------------

def run_model_only(truth):
    """Open-loop: trust the commanded inputs completely, no correction at all."""
    dt, n = truth['dt'], truth['n']
    theta = np.zeros(n)
    v = np.zeros((n, 2))
    p = np.zeros((n, 2))
    for k in range(1, n):
        theta[k] = theta[k - 1] + truth['omega_cmd'][k - 1] * dt
        # commanded acceleration is already expressed in world frame (it's
        # the flight controller's intended direction of travel; the
        # thrust-pulse phase-timing needed to realize it is a separate,
        # lower-level problem assumed already solved), so no rotation is
        # needed to use it directly here.
        a_w = truth['a_cmd'][k - 1]
        v[k] = v[k - 1] + a_w * dt
        p[k] = p[k - 1] + v[k - 1] * dt + 0.5 * a_w * dt ** 2
    return {'theta': theta, 'v': v, 'p': p}


# ----------------------------------------------------------------------
# 4. Extended Kalman Filter: model-based prediction, vision-only correction
# ----------------------------------------------------------------------

class ModelBasedEKF:
    """
    State x = [px, py, vx, vy, theta, spin_bias]  (6,)

    Predict: uses the known commanded omega and commanded world-frame
    acceleration directly (no rotation needed for acceleration, since the
    command is already specified in world frame by the flight controller).
    spin_bias captures the as-yet-unknown gap between commanded and true
    spin rate; it is only observable through its effect on how the vision
    measurement's implied body-frame rotation lines up with the state.

    Correct: body-frame optical flow, same measurement model as v1.
    """

    def __init__(self, dt, q_theta=5e-9, q_bias=2.7e-4, q_vel=1e-8,
                 flow_noise_std=0.06,
                 p0_std=(0.05, 0.05, 0.05, 0.05, 0.05, 0.02)):
        """
        q_bias is the single most important tuning knob in this filter: it
        must be set to roughly match how fast the real spin-rate disturbance
        is expected to drift (wind gusts, motor response lag). Too small,
        and the bias state cannot keep up with a genuinely drifting
        disturbance - once the phase estimate falls behind, every vision
        correction gets applied through the wrong rotation and actively
        corrupts the estimate rather than helping (this is a real failure
        mode this project ran into and fixed, not a hypothetical one). Too
        large, and the filter distrusts its own (already-good) model
        unnecessarily. The default here is tuned for a moderately gusty
        outdoor disturbance level; recalibrate for your actual flight
        environment.
        """
        self.dt = dt
        self.x = np.zeros(6)
        self.P = np.diag(np.array(p0_std) ** 2)
        self.flow_noise_std = flow_noise_std
        self.Q = np.diag([1e-10, 1e-10, q_vel, q_vel, q_theta, q_bias])

    def predict(self, omega_cmd, a_cmd_world):
        dt = self.dt
        px, py, vx, vy, theta, bias = self.x

        omega_hat = omega_cmd + bias
        theta_new = theta + omega_hat * dt

        vx_new = vx + a_cmd_world[0] * dt
        vy_new = vy + a_cmd_world[1] * dt
        px_new = px + vx * dt + 0.5 * a_cmd_world[0] * dt ** 2
        py_new = py + vy * dt + 0.5 * a_cmd_world[1] * dt ** 2

        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = dt  # d(theta_new)/d(bias) - bias ADDS to commanded rate here

        self.x = np.array([px_new, py_new, vx_new, vy_new, theta_new, bias])
        self.P = F @ self.P @ F.T + self.Q

    def update_flow(self, flow_meas):
        px, py, vx, vy, theta, bias = self.x
        c, s = np.cos(theta), np.sin(theta)

        h = np.array([c * vx + s * vy, -s * vx + c * vy])
        H = np.zeros((2, 6))
        H[0, 2], H[0, 3] = c, s
        H[1, 2], H[1, 3] = -s, c
        H[0, 4] = -s * vx + c * vy
        H[1, 4] = -c * vx - s * vy

        Rm = np.eye(2) * self.flow_noise_std ** 2
        y = flow_meas - h
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P


def run_ekf(truth, sensors):
    dt, n = truth['dt'], truth['n']
    ekf = ModelBasedEKF(dt, flow_noise_std=sensors['flow_noise_std'])
    theta = np.zeros(n)
    v = np.zeros((n, 2))
    p = np.zeros((n, 2))
    bias_est = np.zeros(n)

    for k in range(1, n):
        ekf.predict(truth['omega_cmd'][k - 1], truth['a_cmd'][k - 1])
        if sensors['flow_valid'][k]:
            ekf.update_flow(sensors['flow'][k])
        p[k], v[k], theta[k], bias_est[k] = ekf.x[0:2], ekf.x[2:4], ekf.x[4], ekf.x[5]

    return {'theta': theta, 'v': v, 'p': p, 'bias_est': bias_est}


# ----------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------

def position_rmse(truth, est):
    err = est['p'] - truth['p']
    return np.sqrt(np.mean(np.sum(err ** 2, axis=1)))


def final_position_error(truth, est):
    return np.linalg.norm(est['p'][-1] - truth['p'][-1])
