"""
Onboard state estimation for Phantom Twist - v3: closed-loop, grounded in
the published control law.

What changed from v2
---------------------
v2 treated the gap between "commanded" and "true" dynamics as an external,
independent disturbance (wind, generic motor lag). The published paper
(Wang, Yu, Matthews, Alexander, Kriegman, Rubenstein - RSS/arXiv 2605.11296)
describes the REAL control law in Section III-B:

    "The motor executes two power commands per rotation: the average of
    commands regulates vertical thrust, while the difference of commands
    and the switching time regulate the average torque that tilts the
    Phantom Twist and in turn generates horizontal acceleration."

Critically: "synchronization between state estimation and actuation is
critical: a timing error exceeding 5ms can cause loss of control."

This means the controller decides WHEN in the rotation to switch power
levels based on its own phase estimate (theta_est). If that estimate is
wrong, the real-world switching happens at the wrong true phase, and the
resulting net thrust direction is physically rotated away from what was
intended - by approximately the phase estimation error itself. This is a
genuine CLOSED LOOP: the estimator's own error becomes a real disturbance
to the real trajectory, not just a scoring artifact. v3 simulates this
coupling directly instead of injecting an independent random disturbance.

The published vehicle's spin rate (15-25 rps) is a passive aerodynamic
equilibrium (propeller drag torque vs. airframe drag torque, Section II-A),
not something actively steered by this same control law, so spin rate is
still modeled as commanded-plus-independent-disturbance, as in v2.

A second, literature-grounded evaluation metric is added: the paper's own
stated failure threshold is a 5ms actuation timing error. At the vehicle's
spin rate, a given phase error in radians corresponds to a timing error in
seconds (timing_error = phase_error / omega). This lets results be judged
against the paper's own number, rather than an estimated "a few mm" bar.
"""

import numpy as np


def R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class ClosedLoopEKF:
    """Same structure as v2's ModelBasedEKF: state [px,py,vx,vy,theta,spin_bias],
    predicted from commanded (not measured) spin rate and the controller's
    OWN assumed (uncorrupted) thrust direction - the estimator has no way
    to know its own phase error, by definition. Corrected by body-frame
    optical flow only."""

    def __init__(self, dt, q_theta=5e-9, q_bias=2.7e-4, q_vel=1e-8,
                 flow_noise_std=0.06,
                 p0_std=(0.05, 0.05, 0.05, 0.05, 0.05, 0.02)):
        self.dt = dt
        self.x = np.zeros(6)
        self.P = np.diag(np.array(p0_std) ** 2)
        self.flow_noise_std = flow_noise_std
        self.Q = np.diag([1e-10, 1e-10, q_vel, q_vel, q_theta, q_bias])

    def predict(self, omega_cmd, a_assumed_world):
        dt = self.dt
        px, py, vx, vy, theta, bias = self.x
        omega_hat = omega_cmd + bias
        theta_new = theta + omega_hat * dt
        vx_new = vx + a_assumed_world[0] * dt
        vy_new = vy + a_assumed_world[1] * dt
        px_new = px + vx * dt + 0.5 * a_assumed_world[0] * dt ** 2
        py_new = py + vy * dt + 0.5 * a_assumed_world[1] * dt ** 2
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = dt
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


def run_closed_loop(dt, T, spin_rate_rps=20.0, spin_up_time=0.5,
                     F_mag=0.35, desired_omega=0.5,
                     spin_mismatch=0.05, accel_noise_std=0.02,
                     flow_noise_std=0.06, flow_dropout_prob=0.0,
                     use_vision_correction=True, seed=0):
    """
    Runs TRUE dynamics and the EKF together in one causal loop, since the
    true acceleration now depends on the estimator's own current phase
    error (see module docstring). use_vision_correction=False reproduces
    an open-loop estimator (integrates commanded omega forever, no
    correction at all) for comparison - this is the "what if there were no
    state estimation feedback at all" baseline, closer to what actually
    breaks a real system than v2's "model only" baseline was.
    """
    rng = np.random.default_rng(seed)
    n = int(T / dt) + 1
    t = np.linspace(0, T, n)

    omega_target = 2 * np.pi * spin_rate_rps
    omega_cmd = np.where(t < spin_up_time, omega_target * (t / spin_up_time), omega_target)
    spin_disturbance = np.cumsum(rng.normal(0, spin_mismatch * np.sqrt(dt), n))
    omega_true_arr = omega_cmd + spin_disturbance

    desired_angle = desired_omega * t

    ekf = ClosedLoopEKF(dt, flow_noise_std=flow_noise_std)

    theta_true = np.zeros(n)
    v_true = np.zeros((n, 2))
    p_true = np.zeros((n, 2))
    theta_est = np.zeros(n)
    v_est = np.zeros((n, 2))
    p_est = np.zeros((n, 2))
    phase_err_hist = np.zeros(n)
    timing_err_hist = np.zeros(n)  # equivalent actuation timing error, seconds

    flow_valid = rng.random(n) > flow_dropout_prob

    for k in range(1, n):
        # Controller's own belief of its phase (open-loop integration only
        # when vision correction is disabled, to form the no-feedback-at-all
        # baseline; otherwise this is the EKF's current theta estimate).
        th_est_prev = theta_est[k - 1]
        th_true_prev = theta_true[k - 1]
        phase_err = th_est_prev - th_true_prev

        # Realized (true) thrust direction is rotated away from intended by
        # the controller's own phase error - the core closed-loop coupling.
        realized_angle = desired_angle[k - 1] - phase_err
        a_true = F_mag * np.array([np.cos(realized_angle), np.sin(realized_angle)]) \
            + rng.normal(0, accel_noise_std, 2)

        # What the estimator ITSELF assumes it is producing (it cannot know
        # its own phase error) - used only for the EKF's prediction step.
        a_assumed = F_mag * np.array([np.cos(desired_angle[k - 1]), np.sin(desired_angle[k - 1])])

        omega_true = omega_true_arr[k - 1]
        theta_true[k] = th_true_prev + omega_true * dt
        v_true[k] = v_true[k - 1] + a_true * dt
        p_true[k] = p_true[k - 1] + v_true[k - 1] * dt + 0.5 * a_true * dt ** 2

        if use_vision_correction:
            ekf.predict(omega_cmd[k - 1], a_assumed)
            flow_body_true = R(-theta_true[k]) @ v_true[k]
            flow_meas = flow_body_true + rng.normal(0, flow_noise_std, 2)
            if flow_valid[k]:
                ekf.update_flow(flow_meas)
            p_est[k], v_est[k], theta_est[k] = ekf.x[0:2], ekf.x[2:4], ekf.x[4]
        else:
            # No correction at all: pure open-loop phase integration.
            theta_est[k] = th_est_prev + omega_cmd[k - 1] * dt
            v_est[k] = v_est[k - 1] + a_assumed * dt
            p_est[k] = p_est[k - 1] + v_est[k - 1] * dt + 0.5 * a_assumed * dt ** 2

        phase_err_hist[k] = phase_err
        timing_err_hist[k] = phase_err / max(omega_true, 1e-6)

    return {
        't': t, 'dt': dt,
        'theta_true': theta_true, 'p_true': p_true, 'v_true': v_true,
        'theta_est': theta_est, 'p_est': p_est, 'v_est': v_est,
        'phase_err': phase_err_hist, 'timing_err': timing_err_hist,
    }


def position_rmse(result):
    err = result['p_est'] - result['p_true']
    return np.sqrt(np.mean(np.sum(err ** 2, axis=1)))


def final_position_error(result):
    return np.linalg.norm(result['p_est'][-1] - result['p_true'][-1])


def fraction_over_timing_threshold(result, threshold_s=0.005, settle_frac=0.1):
    """Fraction of (post-settling) time the equivalent actuation timing
    error exceeds the paper's own stated loss-of-control threshold (5ms)."""
    n = len(result['t'])
    start = int(n * settle_frac)
    over = np.abs(result['timing_err'][start:]) > threshold_s
    return np.mean(over)
