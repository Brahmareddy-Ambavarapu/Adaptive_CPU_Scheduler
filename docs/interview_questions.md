# Interview Questions & Answers

**Q: Why use Machine Learning for CPU scheduling instead of traditional heuristics?**
**A:** Traditional heuristics (like always defaulting to Round Robin) are a compromise. They handle average workloads decently but fail spectacularly on edge cases (like using FCFS for thousands of tiny jobs, causing convoy effects). ML allows the system to recognize the specific statistical signature of an incoming workload (e.g. Bimodal or PriorityHeavy) and dynamically deploy the algorithm best suited for that exact moment.

**Q: Why use classification instead of regression to predict specific metric times?**
**A:** Because absolute metric values (like predicting exactly 142ms of waiting time) fluctuate wildly based on scale and hardware speed. Classification simplifies the problem: we just need the model to point to the algorithm name that minimizes the loss function relative to its peers.

**Q: Why use Random Forest?**
**A:** Random Forest is an ensemble tree method that excels at capturing non-linear boundaries without heavy parameter tuning. Workload scheduling boundaries are highly non-linear (e.g. if arrival rate is low, average burst doesn't matter much, but if arrival rate is high, average burst dictates the entire queue dynamic). It naturally prevents overfitting compared to a single deep Decision Tree.

**Q: Why perform workload-level dataset splitting instead of process-level?**
**A:** Because the algorithms schedule *groups* of processes. If we split at the process level, we would artificially tear apart the statistical structure of a workload and cause data leakage, where parts of a single workload exist in both train and test sets, violating the independent and identically distributed (i.i.d) assumption.

**Q: What is Target Leakage and how did you prevent it?**
**A:** Target leakage occurs when the model uses information during training that won't be available at prediction time in production. We prevented this by strictly computing our 10 features from *pre-execution* state (arrival rates, burst times). Post-execution metrics like waiting time or context switches are strictly excluded from the ML input. Furthermore, our StandardScaler is fitted *only* on the training set.

**Q: Why do you normalize the metrics before computing the Objective Score?**
**A:** Because different metrics exist on different scales. Context switches might range from 0 to 50, whereas Turnaround Time might range from 50 to 5000. If we just added them, Turnaround Time would dominate the objective score. We normalize each algorithm's metric against the maximum observed value for that specific workload, bounding everything between 0.0 and 1.0.

**Q: What is the "Oracle" and what is Performance Regret?**
**A:** The Oracle is an un-deployable, perfect hindsight baseline that evaluates all 4 algorithms concurrently on identical workload copies and mathematically chooses the winner. Performance Regret is the percentage deviation of our ML model's choice from that true Oracle optimum. Even if the ML misclassifies, if the Regret is 0.5%, the practical hardware loss is negligible.

**Q: Why do classification accuracy and regret diverge?**
**A:** Multiple schedulers can perform similarly well on certain workloads (e.g. FCFS and SJF on a workload where all jobs have identical burst times). If the model guesses SJF but the Oracle defaults to FCFS based on tie-breaking, the classification accuracy drops, but the Regret remains exactly 0%.

**Q: How does the ML adaptive scheduler work at runtime?**
**A:** Before a batch of processes starts, the C++ engine extracts 10 features in $O(N)$ time. It serializes these features and queries the trained `scikit-learn` model via a lightweight JSON pipe. The model returns a single scheduler name. C++ instantiates that specific base scheduler and executes the workload seamlessly.

**Q: What does the Generalization experiment prove?**
**A:** By generating a brand-new dataset with an unseen seed (2026), we prove the model actually learned the fundamental physics of queuing theory rather than simply memorizing the specific arrays in the training set.

**Q: What does Feature Ablation show?**
**A:** It shows which features carry the most weight. Retraining the model while intentionally dropping one feature at a time reveals that removing features like `short_job_ratio` severely increases Regret, meaning the model relies on it heavily to make optimal decisions.

**Q: How would you deploy this in a real OS?**
**A:** A real OS (like Linux CFS) uses a runqueue and schedules dynamically. This project operates on batch evaluation. To adapt it to a real kernel, the ML inference would need to be rewritten in raw C or eBPF to prevent Python interpreter overhead, and it would trigger at specific epochs (e.g. every 50ms tick) to evaluate the current runqueue snapshot and adjust weights, rather than swapping the entire algorithm outright.
