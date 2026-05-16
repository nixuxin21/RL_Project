Archived exploratory experiments.

These scripts are kept for reproducibility of earlier research branches, but
they are not the current main line. The active limited-CSI direction is driven
from the repository root by `evaluate_limited_csi_ms_aircomp.py`.

Some deprecated directions still live at the repository root because they are
large scripts referenced by reports, Makefile smoke targets, or historical
commands. They are logically archived in `docs/DEPRECATED_DIRECTIONS.md` even
when the physical file has not been moved.

- `evaluate_noisy_feature_sweep.py`: robustness of exact codebook quality
  features under synthetic feature noise.
- `train_learned_probing_selector.py`: supervised learned top-B probing
  selector from low-dimensional state.
- `evaluate_probing_cost_tradeoff.py`: post-hoc utility analysis for older
  partial/learned probing summaries.
