import json
import numpy as np
import matplotlib.pyplot as plt

with open('/home/claude/v3/v3_results.json') as f:
    results = json.load(f)

levels = [0.05, 0.15, 0.3, 0.5]
on_final = [results[str(l)]['on_final_mean'] for l in levels]
on_final_std = [results[str(l)]['on_final_std'] for l in levels]
off_final = [results[str(l)]['off_final_mean'] for l in levels]
off_final_std = [results[str(l)]['off_final_std'] for l in levels]
on_frac = [results[str(l)]['on_frac_mean'] for l in levels]
off_frac = [results[str(l)]['off_frac_mean'] for l in levels]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.errorbar(levels, off_final, yerr=off_final_std, marker='o', color='#D85A30',
            capsize=4, label='No correction (open loop)')
ax.errorbar(levels, on_final, yerr=on_final_std, marker='o', color='#1D9E75',
            capsize=4, label='EKF (vision-corrected)')
ax.set_yscale('log')
ax.set_xlabel('spin-rate disturbance level')
ax.set_ylabel('final position error (m, log scale)')
ax.set_title('Position error: closed-loop v3')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')

ax = axes[1]
width = 0.015
x = np.array(levels)
ax.bar(x - width/2, off_frac, width=width*0.9, color='#D85A30', label='No correction')
ax.bar(x + width/2, on_frac, width=width*0.9, color='#1D9E75', label='EKF (vision-corrected)')
for xi, v in zip(x, off_frac):
    ax.text(xi - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
for xi in x:
    ax.text(xi + width/2, 0.02, '0.00', ha='center', fontsize=9, color='#1D9E75')
ax.set_xlabel('spin-rate disturbance level')
ax.set_ylabel("fraction of flight time over paper's 5ms\ntiming-error loss-of-control threshold")
ax.set_title("Time spent beyond the paper's own stated failure threshold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/v3/v3_results.png', dpi=150)
print("saved v3_results.png")
