import numpy as np
from phantom_twist_v3 import run_closed_loop, final_position_error, fraction_over_timing_threshold

dt = 0.001
T = 15.0
N_SEEDS = 20
levels = [0.05, 0.15, 0.3, 0.5]

results = {}
for spin_mm in levels:
    on_final, off_final, on_frac, off_frac = [], [], [], []
    for seed in range(N_SEEDS):
        r_on = run_closed_loop(dt, T, spin_mismatch=spin_mm, use_vision_correction=True, seed=seed)
        r_off = run_closed_loop(dt, T, spin_mismatch=spin_mm, use_vision_correction=False, seed=seed)
        on_final.append(final_position_error(r_on))
        off_final.append(final_position_error(r_off))
        on_frac.append(fraction_over_timing_threshold(r_on))
        off_frac.append(fraction_over_timing_threshold(r_off))
    results[spin_mm] = {
        'on_final_mean': np.mean(on_final), 'on_final_std': np.std(on_final),
        'off_final_mean': np.mean(off_final), 'off_final_std': np.std(off_final),
        'on_frac_mean': np.mean(on_frac), 'off_frac_mean': np.mean(off_frac),
    }
    print(f"spin_mismatch={spin_mm}")
    print(f"  final error   WITH={np.mean(on_final):.3f}+-{np.std(on_final):.3f}  WITHOUT={np.mean(off_final):.3f}+-{np.std(off_final):.3f}")
    print(f"  frac>5ms      WITH={np.mean(on_frac):.3f}  WITHOUT={np.mean(off_frac):.3f}")

import json
with open('/home/claude/v3/v3_results.json', 'w') as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
print("\nsaved v3_results.json")
