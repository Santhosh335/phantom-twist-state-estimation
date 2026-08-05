import numpy as np
import matplotlib.pyplot as plt
from phantom_twist_ekf import (simulate_ground_truth, generate_sensors,
                                run_dead_reckoning, run_ekf, final_position_error)

dt = 0.001
T = 8.0
N_SEEDS = 15
dropout_rates = [0.0, 0.2, 0.4, 0.6, 0.8]

truth = simulate_ground_truth(dt, T, spin_rate_rps=20.0)

dr_final = final_position_error(truth, run_dead_reckoning(
    truth, generate_sensors(truth, gyro_bias=0.05, seed=0)))

ekf_means, ekf_stds = [], []
for dp in dropout_rates:
    finals = []
    for seed in range(N_SEEDS):
        sensors = generate_sensors(truth, gyro_bias=0.05, flow_dropout_prob=dp, seed=seed)
        ek = run_ekf(truth, sensors)
        finals.append(final_position_error(truth, ek))
    ekf_means.append(np.mean(finals))
    ekf_stds.append(np.std(finals))
    print(f"flow dropout {dp*100:>3.0f}%:  EKF final error = {np.mean(finals):.3f} ± {np.std(finals):.3f} m")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar([d*100 for d in dropout_rates], ekf_means, yerr=ekf_stds,
            marker='o', color='#1D9E75', capsize=4, label='EKF (IMU + flow)')
ax.axhline(dr_final, color='#D85A30', linestyle='--', label='Dead reckoning (no flow at all)')
ax.set_xlabel('optical-flow dropout rate (%)')
ax.set_ylabel('final position error (m)')
ax.set_title(f'EKF robustness to flow sensor dropout (n={N_SEEDS} seeds/point)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('robustness.png', dpi=150)
print("\nsaved robustness.png")
