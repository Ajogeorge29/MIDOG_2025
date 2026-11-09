# Notebooks

Keep notebooks thin: put reusable loading, validation, training, and inference
logic in `src/midog2025/`, then import it here. This prevents notebook-only
results and makes experiments testable.

- `MitoGuard_Colab_Training.ipynb`: Colab setup, persistent Drive data wiring,
  audit gates, bounded pilot execution, and main-training readiness manifest.
