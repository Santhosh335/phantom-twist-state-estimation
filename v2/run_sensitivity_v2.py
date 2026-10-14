import numpy as np
import matplotlib.pyplot as plt
from phantom_twist_ekf_v2 import (simulate_ground_truth, generate_flow_sensor,
                                   run_model_only, run_ekf, final_position_error)

dt = 0.001
T = 8.0
N_SEEDS = 15
# scale spin_mismatch and accel_mismatch_frac together as a single
# "disturbance level" knob, from calm to very gusty.
levels = [0.02, 0.1, 0.2, 0.3, 0.45, 0.6]

mo_means, mo_stds, ek_means, ek_stds = [], [], [], []
for lvl in levels:
    mo_f, ek_f = [], []
    for seed in range(N_SEEDS):
        truth = simulate_ground_truth(dt, T, spin_rate_rps=20.0,
                                       spin_mismatch=lvl, accel_mismatch_frac=lvl, seed=seed)
        flow = generate_flow_sensor(truth, seed=seed)
        mo_f.append(final_position_error(truth, run_model_only(truth)))
        ek_f.append(final_position_error(truth, run_ekf(truth, flow)))
    mo_means.append(np.mean(mo_f)); mo_stds.append(np.std(mo_f))
    ek_means.append(np.mean(ek_f)); ek_stds.append(np.std(ek_f))
    print(f"level={lvl:.2f}  model-only={np.mean(mo_f):.3f}+-{np.std(mo_f):.3f}  "
          f"EKF={np.mean(ek_f):.3f}+-{np.std(ek_f):.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.errorbar(levels, mo_means, yerr=mo_stds, marker='o', color='#D85A30',
            capsize=4, label='Model only (open loop)')
ax.errorbar(levels, ek_means, yerr=ek_stds, marker='o', color='#1D9E75',
            capsize=4, label='EKF (model + vision)')
ax.set_xlabel('disturbance level (spin-rate + thrust mismatch)')
ax.set_ylabel('final position error (m)')
ax.set_title(f'Where vision correction starts to pay off (n={N_SEEDS} seeds/point)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/v2/sensitivity_v2.png', dpi=150)
print("\nsaved sensitivity_v2.png")
