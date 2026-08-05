import numpy as np
import matplotlib.pyplot as plt
from phantom_twist_ekf import (simulate_ground_truth, generate_sensors,
                                run_dead_reckoning, run_ekf, position_rmse,
                                final_position_error)

dt = 0.001
T = 8.0
truth = simulate_ground_truth(dt, T, spin_rate_rps=20.0)
sensors = generate_sensors(truth, gyro_bias=0.05, seed=1)

dr = run_dead_reckoning(truth, sensors)
ek = run_ekf(truth, sensors)

print(f"Dead reckoning  RMSE: {position_rmse(truth, dr):.3f} m   final error: {final_position_error(truth, dr):.3f} m")
print(f"EKF             RMSE: {position_rmse(truth, ek):.3f} m   final error: {final_position_error(truth, ek):.3f} m")
print(f"True gyro bias: {sensors['gyro_bias_true']:.4f} rad/s   EKF estimated (final): {ek['bias_est'][-1]:.4f} rad/s")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(truth['p'][:, 0], truth['p'][:, 1], 'k-', lw=2, label='Ground truth')
ax.plot(dr['p'][:, 0], dr['p'][:, 1], '--', color='#D85A30', lw=1.3, label='Dead reckoning (IMU only)')
ax.plot(ek['p'][:, 0], ek['p'][:, 1], '-', color='#1D9E75', lw=1.5, label='EKF (IMU + optical flow)')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title('Trajectory: ground truth vs estimators')
ax.legend(fontsize=8); ax.axis('equal'); ax.grid(alpha=0.3)

ax = axes[1]
err_dr = np.linalg.norm(dr['p'] - truth['p'], axis=1)
err_ek = np.linalg.norm(ek['p'] - truth['p'], axis=1)
ax.plot(truth['t'], err_dr, color='#D85A30', label='Dead reckoning')
ax.plot(truth['t'], err_ek, color='#1D9E75', label='EKF')
ax.set_xlabel('time (s)'); ax.set_ylabel('position error (m)')
ax.set_title('Position error over time')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('single_run.png', dpi=150)
print("saved single_run.png")
