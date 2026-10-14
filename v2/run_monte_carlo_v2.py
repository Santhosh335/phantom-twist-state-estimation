import numpy as np
import matplotlib.pyplot as plt
from phantom_twist_ekf_v2 import (simulate_ground_truth, generate_flow_sensor,
                                   run_model_only, run_ekf, position_rmse,
                                   final_position_error)

dt = 0.001
T = 8.0
N_SEEDS = 30
# low disturbance ("calm") vs realistic outdoor ("gusty") - the EKF's
# value proposition should show up specifically in the gusty case.
SCENARIOS = [
    ('calm', 0.02, 0.05),
    ('gusty', 0.3, 0.4),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for i, (label, spin_mm, acc_mm) in enumerate(SCENARIOS):
    mo_final, ek_final = [], []
    for seed in range(N_SEEDS):
        truth = simulate_ground_truth(dt, T, spin_rate_rps=20.0,
                                       spin_mismatch=spin_mm, accel_mismatch_frac=acc_mm, seed=seed)
        flow = generate_flow_sensor(truth, seed=seed)
        mo = run_model_only(truth)
        ek = run_ekf(truth, flow)
        mo_final.append(final_position_error(truth, mo))
        ek_final.append(final_position_error(truth, ek))

    mo_final, ek_final = np.array(mo_final), np.array(ek_final)
    wins = np.sum(ek_final < mo_final)
    print(f"{label}: model-only final = {mo_final.mean():.3f} +- {mo_final.std():.3f} m   "
          f"EKF final = {ek_final.mean():.3f} +- {ek_final.std():.3f} m   "
          f"EKF wins {wins}/{N_SEEDS}")

    ax = axes[i]
    bp = ax.boxplot([mo_final, ek_final], tick_labels=['Model only', 'EKF'],
                     patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], ['#F0997B', '#5DCAA5']):
        patch.set_facecolor(color)
    ax.set_ylabel('final position error (m)')
    ax.set_title(f'{label} disturbance ({N_SEEDS} seeds)')
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/v2/monte_carlo_v2.png', dpi=150)
print("\nsaved monte_carlo_v2.png")
