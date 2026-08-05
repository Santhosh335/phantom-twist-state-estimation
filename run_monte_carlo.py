import numpy as np
import matplotlib.pyplot as plt
from phantom_twist_ekf import (simulate_ground_truth, generate_sensors,
                                run_dead_reckoning, run_ekf, position_rmse,
                                final_position_error)

dt = 0.001
T = 8.0
N_SEEDS = 30

truth = simulate_ground_truth(dt, T, spin_rate_rps=20.0)

dr_rmse, ek_rmse = [], []
dr_final, ek_final = [], []

for seed in range(N_SEEDS):
    sensors = generate_sensors(truth, gyro_bias=0.05, seed=seed)
    dr = run_dead_reckoning(truth, sensors)
    ek = run_ekf(truth, sensors)
    dr_rmse.append(position_rmse(truth, dr))
    ek_rmse.append(position_rmse(truth, ek))
    dr_final.append(final_position_error(truth, dr))
    ek_final.append(final_position_error(truth, ek))

dr_rmse, ek_rmse = np.array(dr_rmse), np.array(ek_rmse)
dr_final, ek_final = np.array(dr_final), np.array(ek_final)

print(f"N = {N_SEEDS} seeds, spin rate = 20 rev/s, gyro bias = 0.05 rad/s, T = {T}s\n")
print(f"{'':22s}{'RMSE (m)':>18s}{'Final error (m)':>20s}")
print(f"{'Dead reckoning':22s}{dr_rmse.mean():>10.3f} ± {dr_rmse.std():<6.3f}{dr_final.mean():>12.3f} ± {dr_final.std():<6.3f}")
print(f"{'EKF':22s}{ek_rmse.mean():>10.3f} ± {ek_rmse.std():<6.3f}{ek_final.mean():>12.3f} ± {ek_final.std():<6.3f}")
print(f"\nMean RMSE reduction:  {(1 - ek_rmse.mean()/dr_rmse.mean())*100:.1f}%")
print(f"Mean final-error reduction: {(1 - ek_final.mean()/dr_final.mean())*100:.1f}%")

# paired comparison: EKF should beat dead reckoning on (almost) every seed,
# not just on average
wins = np.sum(ek_final < dr_final)
print(f"EKF beats dead reckoning on final error in {wins}/{N_SEEDS} seeds")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ax = axes[0]
bp = ax.boxplot([dr_rmse, ek_rmse], tick_labels=['Dead reckoning', 'EKF'],
                patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], ['#F0997B', '#5DCAA5']):
    patch.set_facecolor(color)
ax.set_ylabel('position RMSE over trajectory (m)')
ax.set_title(f'RMSE across {N_SEEDS} Monte Carlo seeds')
ax.grid(alpha=0.3, axis='y')

ax = axes[1]
bp = ax.boxplot([dr_final, ek_final], tick_labels=['Dead reckoning', 'EKF'],
                patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], ['#F0997B', '#5DCAA5']):
    patch.set_facecolor(color)
ax.set_ylabel('final position error (m)')
ax.set_title(f'Final error across {N_SEEDS} seeds')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('monte_carlo.png', dpi=150)
print("\nsaved monte_carlo.png")
