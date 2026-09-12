#ifndef WORKLOAD_FEATURES_HPP
#define WORKLOAD_FEATURES_HPP

struct WorkloadFeatures {
    int num_processes = 0;
    double average_burst = 0.0;
    double burst_std = 0.0;
    int min_burst = 0;
    int max_burst = 0;
    double arrival_rate = 0.0;
    double average_priority = 0.0;
    double priority_std = 0.0;
    double short_job_ratio = 0.0;
    double long_job_ratio = 0.0;
};

#endif // WORKLOAD_FEATURES_HPP
