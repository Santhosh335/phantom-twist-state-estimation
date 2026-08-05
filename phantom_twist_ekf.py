"""
Onboard state estimation for a Phantom-Twist-style spinning UAV.

Problem setup
-------------
Phantom Twist (Northwestern, RSS 2026) has one motor/propeller spinning
one way while the *entire airframe* (batteries, PCB, counterweight, and
any onboard sensors) counter-rotates the opposite way at 15-25 rev/s.
The published prototype relies on an external motion-capture rig for
localization. This module simulates a candidate onboard alternative:
fuse a rigidly-mounted rate gyro + accelerometer with a rigidly-mounted
optical-flow sensor, all of which see the world through a frame that is
itself spinning at up to ~160 rad/s.

Two estimators are compared:
  1. Naive dead reckoning  - integrate gyro + accelerometer only.
  2. EKF (this work)       - same IMU prediction, corrected by fusing
                              body-frame optical flow measurements.

Coordinate convention: 2D world frame (x, y). Body frame is world frame
rotated by theta(t) (the spin angle). R(theta) maps body -> world.
"""

import numpy as np


def R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------
# 1. Ground truth trajectory
# ----------------------------------------------------------------------

def simulate_ground_truth(dt, T, spin_rate_rps=20.0, spin_up_time=0.5,
                           accel_amp=0.35, accel_omega=0.5):
    """
    Generates the true trajectory of the drone's center of mass and its
    true spin angle. Spin rate ramps from 0 to spin_rate_rps (rev/s) over
    spin_up_time seconds, then holds steady - matching how Phantom Twist
    needs an external launcher to reach operating speed before free flight.

    Translational motion is driven by a smooth world-frame acceleration
    profile, standing in for the thrust-pulse-timing controller described
    in the paper (position/velocity control by varying motor thrust at
    specific points in each rotation) without needing to simulate that
    controller explicitly.
    """
    n = int(T / dt) + 1
    t = np.linspace(0, T, n)

    omega_target = 2 * np.pi * spin_rate_rps
    omega_true = np.where(t < spin_up_time, omega_target * (t / spin_up_time), omega_target)
    theta_true = np.concatenate(([0.0], np.cumsum((omega_true[:-1] + omega_true[1:]) / 2 * dt)))

    a_world = np.stack([
        accel_amp * np.cos(accel_omega * t),
        accel_amp * np.sin(accel_omega * t)
    ], axis=1)

    v_world = np.zeros((n, 2))
    p_world = np.zeros((n, 2))
    for k in range(1, n):
        v_world[k] = v_world[k - 1] + a_world[k - 1] * dt
        p_world[k] = p_world[k - 1] + v_world[k - 1] * dt + 0.5 * a_world[k - 1] * dt ** 2

    return {
        't': t, 'dt': dt, 'n': n,
        'theta': theta_true, 'omega': omega_true,
        'p': p_world, 'v': v_world, 'a': a_world,
    }


# ----------------------------------------------------------------------
# 2. Sensor simulation (all sensors rigidly mounted to the spinning body)
# ----------------------------------------------------------------------

def generate_sensors(truth, gyro_bias=0.05, gyro_noise_std=0.02,
                      accel_noise_std=0.05, flow_noise_std=0.06,
                      flow_dropout_prob=0.0, seed=0):
    """
    gyro_bias        : constant unmodeled bias, rad/s (typical cheap MEMS gyro)
    gyro_noise_std    : rad/s, white noise
    accel_noise_std   : m/s^2, white noise on body-frame accelerometer
    flow_noise_std    : m/s, white noise on body-frame optical-flow velocity
    flow_dropout_prob : fraction of steps where the flow sensor has no valid
                         reading (motion blur / low texture / glare) - set >0
                         to stress-test the filter.
    """
    rng = np.random.default_rng(seed)
    n = truth['n']

    gyro_meas = truth['omega'] + gyro_bias + rng.normal(0, gyro_noise_std, n)

    accel_body_true = np.einsum('nij,nj->ni',
                                 np.stack([R(-th) for th in truth['theta']]),
                                 truth['a'])
    accel_meas = accel_body_true + rng.normal(0, accel_noise_std, (n, 2))

    flow_body_true = np.einsum('nij,nj->ni',
                                np.stack([R(-th) for th in truth['theta']]),
                                truth['v'])
    flow_meas = flow_body_true + rng.normal(0, flow_noise_std, (n, 2))
    flow_valid = rng.random(n) > flow_dropout_prob

    return {
        'gyro': gyro_meas, 'accel': accel_meas,
        'flow': flow_meas, 'flow_valid': flow_valid,
        'gyro_bias_true': gyro_bias,
        'gyro_noise_std': gyro_noise_std, 'accel_noise_std': accel_noise_std,
        'flow_noise_std': flow_noise_std,
    }


# ----------------------------------------------------------------------
# 3. Naive dead reckoning (baseline)
# ----------------------------------------------------------------------

def run_dead_reckoning(truth, sensors):
    dt, n = truth['dt'], truth['n']
    theta = np.zeros(n)
    v = np.zeros((n, 2))
    p = np.zeros((n, 2))
    for k in range(1, n):
        theta[k] = theta[k - 1] + sensors['gyro'][k - 1] * dt
        a_w = R(theta[k - 1]) @ sensors['accel'][k - 1]
        v[k] = v[k - 1] + a_w * dt
        p[k] = p[k - 1] + v[k - 1] * dt + 0.5 * a_w * dt ** 2
    return {'theta': theta, 'v': v, 'p': p}


# ----------------------------------------------------------------------
# 4. Extended Kalman Filter
# ----------------------------------------------------------------------

class SpinningBodyEKF:
    """
    State x = [px, py, vx, vy, theta, gyro_bias]  (6,)

    Prediction uses the IMU (gyro + body-frame accelerometer) as control
    inputs (strapdown-INS style). Correction uses body-frame optical flow,
    which is the only aiding sensor available onboard once mocap is removed.
    """

    def __init__(self, dt, q_theta=1e-9, q_bias=1e-12, q_vel=1e-7,
                 accel_noise_std=0.05, flow_noise_std=0.06,
                 gyro_noise_std=0.02,
                 p0_std=(0.05, 0.05, 0.05, 0.05, 0.05, 0.02)):
        self.dt = dt
        self.x = np.zeros(6)
        self.P = np.diag(np.array(p0_std) ** 2)
        self.accel_noise_std = accel_noise_std
        self.flow_noise_std = flow_noise_std
        self.gyro_noise_std = gyro_noise_std
        # process noise (per-step, tuned for dt ~1ms): a small residual
        # random-walk on top of what the IMU noise (propagated through G,
        # Rn below) already injects. These stay small on purpose - most
        # of the real uncertainty growth should come from actual sensor
        # noise, not from an arbitrarily large Q papering over bugs.
        self.Q = np.diag([1e-10, 1e-10, q_vel, q_vel, q_theta, q_bias])

    def predict(self, gyro_meas, accel_meas):
        dt = self.dt
        px, py, vx, vy, theta, bg = self.x

        omega_hat = gyro_meas - bg
        theta_new = theta + omega_hat * dt

        c, s = np.cos(theta), np.sin(theta)
        ax_b, ay_b = accel_meas
        a_world = np.array([c * ax_b - s * ay_b, s * ax_b + c * ay_b])

        vx_new = vx + a_world[0] * dt
        vy_new = vy + a_world[1] * dt
        px_new = px + vx * dt + 0.5 * a_world[0] * dt ** 2
        py_new = py + vy * dt + 0.5 * a_world[1] * dt ** 2

        # Jacobian of f wrt state
        da_dtheta = np.array([-s * ax_b - c * ay_b, c * ax_b - s * ay_b])
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2, 4] = da_dtheta[0] * dt
        F[3, 4] = da_dtheta[1] * dt
        F[0, 4] = 0.5 * da_dtheta[0] * dt ** 2
        F[1, 4] = 0.5 * da_dtheta[1] * dt ** 2
        F[4, 5] = -dt  # d(theta_new)/d(bias)

        # accelerometer + gyro noise injected through the input Jacobian G
        G = np.zeros((6, 3))  # noise inputs: [n_ax, n_ay, n_gyro]
        G[2, 0], G[3, 1] = dt, dt
        G[0, 0], G[1, 1] = 0.5 * dt ** 2, 0.5 * dt ** 2
        G[4, 2] = -dt
        Rn = np.diag([self.accel_noise_std ** 2, self.accel_noise_std ** 2,
                      self.gyro_noise_std ** 2])

        self.x = np.array([px_new, py_new, vx_new, vy_new, theta_new, bg])
        self.P = F @ self.P @ F.T + G @ Rn @ G.T + self.Q

    def update_flow(self, flow_meas):
        px, py, vx, vy, theta, bg = self.x
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
    ekf = SpinningBodyEKF(dt, accel_noise_std=sensors['accel_noise_std'],
                           flow_noise_std=sensors['flow_noise_std'],
                           gyro_noise_std=sensors['gyro_noise_std'])
    theta = np.zeros(n)
    v = np.zeros((n, 2))
    p = np.zeros((n, 2))
    bias_est = np.zeros(n)

    for k in range(1, n):
        ekf.predict(sensors['gyro'][k - 1], sensors['accel'][k - 1])
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
