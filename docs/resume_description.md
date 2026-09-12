# Resume Description

**ML-Based Adaptive CPU Scheduling Framework**

*   Designed and implemented a custom C++ CPU scheduling simulator that evaluates machine-learning-driven adaptive scheduling strategies against an empirically perfect Oracle.
*   Engineered a full Python ML pipeline (`scikit-learn`, `pandas`) utilizing a Random Forest Classifier to infer the optimal algorithm (FCFS, SJF, RR, Priority) purely from pre-execution workload statistical features.
*   Achieved an average scheduling Regret of just **3.5%** against the true Oracle optimum, significantly outperforming hardcoded heuristic schedulers and proving dynamic algorithm switching feasibility.

**Technology Stack:** C++17, Python 3, scikit-learn, Pandas, Matplotlib, PyTest.

### Research Contribution
Demonstrated that computationally inexpensive, aggregate statistical workload features ($O(N)$ extraction) are sufficient for a machine-learning classification model to effectively predict the optimal CPU scheduling algorithm. By shifting from a static heuristic to a dynamic, workload-aware prediction model, the system avoids "catastrophic" scheduling mismatches (e.g., using FCFS on a short-job heavy batch) and maintains near-optimal hardware utilization.
