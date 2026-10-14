import numpy as np
import matplotlib.pyplot as plt
from phantom_twist_ekf_v2 import (simulate_ground_truth, generate_flow_sensor,
                                   run_model_only, run_ekf, position_rmse,
                                   final_position_error)

dt = 0.001
T = 8.0
truth = simulate_ground_truth(dt, T, spin_rate_rps=20.0,
                               spin_mismatch=0.3, accel_mismatch_frac=0.4, seed=2)
flow = generate_flow_sensor(truth, seed=2)

mo = run_model_only(truth)
ek = run_ekf(truth, flow)

print(f"Model-only (open loop)  RMSE: {position_rmse(truth, mo):.3f} m   final error: {final_position_error(truth, mo):.3f} m")
print(f"EKF (model + vision)    RMSE: {position_rmse(truth, ek):.3f} m   final error: {final_position_error(truth, ek):.3f} m")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(truth['p'][:, 0], truth['p'][:, 1], 'k-', lw=2, label='Ground truth')
ax.plot(mo['p'][:, 0], mo['p'][:, 1], '--', color='#D85A30', lw=1.3, label='Model only (open loop)')
ax.plot(ek['p'][:, 0], ek['p'][:, 1], '-', color='#1D9E75', lw=1.5, label='EKF (model + vision)')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title('Trajectory: no onboard gyro, vision-only correction')
ax.legend(fontsize=8); ax.axis('equal'); ax.grid(alpha=0.3)

ax = axes[1]
err_mo = np.linalg.norm(mo['p'] - truth['p'], axis=1)
err_ek = np.linalg.norm(ek['p'] - truth['p'], axis=1)
ax.plot(truth['t'], err_mo, color='#D85A30', label='Model only')
ax.plot(truth['t'], err_ek, color='#1D9E75', label='EKF')
ax.set_xlabel('time (s)'); ax.set_ylabel('position error (m)')
ax.set_title('Position error (gusty disturbance scenario)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/v2/single_run_v2.png', dpi=150)
print("saved single_run_v2.png")
