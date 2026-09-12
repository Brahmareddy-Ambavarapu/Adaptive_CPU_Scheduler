# Project Status

**Steps 1–16 are COMPLETE.**

## Architecture & Schedulers
- C++ Simulation Engine (Ticks, Process execution, idle handling) is fully robust.
- FCFS, Non-Preemptive SJF, Preemptive Round Robin, and Non-Preemptive Priority schedulers are complete and mathematically verified.
- The `OracleScheduler` correctly implements the 4-way evaluation.

## ML Pipeline
- `WorkloadFeatureExtractor` accurately dumps $O(N)$ statistical vectors in both C++ and Python.
- Preprocessing, scaling, label mapping, and Random Forest training are completely deterministic.
- Model artifacts (`best_model.joblib`, `scaler.joblib`, `label_mapping.json`) are locked and ready for inference.

## Experiments & Reproducibility
- All 4 sensitivity experiments (Generalization, Ablation, Objective Weights, Training Size) execute flawlessly via `experiments/run_step15.py`.
- CSV outputs and PNG visualizations are mapped automatically to `results/`.
- Cross-platform C++ via `g++` and Python `subprocess.run` is fully verified on Windows.

## Testing
- PyTest validation suite is passing 25/25 tests (zero failures).
- Memory footprint is clean. No `std::popen` hangs exist.

## Known Limitations
- The C++ ML bridge relies on temporary JSON files (`ml_inference_temp_<timestamp>_<rand>.json`). Parallel execution of `scheduler.exe` instances uses timestamps to prevent file locking collisions on Windows.
- The simulation models purely CPU-bound tasks without blocking I/O constraints.
