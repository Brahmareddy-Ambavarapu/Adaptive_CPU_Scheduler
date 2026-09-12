#include "workload_feature_extractor.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

WorkloadFeatures WorkloadFeatureExtractor::extract(const std::vector<Process>& processes) {
    WorkloadFeatures features;
    
    if (processes.empty()) {
        return features;
    }
    
    features.num_processes = processes.size();
    
    long long total_burst = 0;
    long long total_priority = 0;
    int min_arrival = std::numeric_limits<int>::max();
    int max_arrival = std::numeric_limits<int>::min();
    
    features.min_burst = std::numeric_limits<int>::max();
    features.max_burst = std::numeric_limits<int>::min();
    
    for (const auto& p : processes) {
        total_burst += p.burst_time;
        total_priority += p.priority;
        
        features.min_burst = std::min(features.min_burst, p.burst_time);
        features.max_burst = std::max(features.max_burst, p.burst_time);
        
        min_arrival = std::min(min_arrival, p.arrival_time);
        max_arrival = std::max(max_arrival, p.arrival_time);
    }
    
    features.average_burst = static_cast<double>(total_burst) / features.num_processes;
    features.average_priority = static_cast<double>(total_priority) / features.num_processes;
    
    double burst_variance_sum = 0.0;
    double priority_variance_sum = 0.0;
    int short_jobs = 0;
    int long_jobs = 0;
    
    for (const auto& p : processes) {
        double burst_diff = p.burst_time - features.average_burst;
        burst_variance_sum += burst_diff * burst_diff;
        
        double prio_diff = p.priority - features.average_priority;
        priority_variance_sum += prio_diff * prio_diff;
        
        if (p.burst_time <= features.average_burst) {
            short_jobs++;
        } else {
            long_jobs++;
        }
    }
    
    // Population standard deviation (divide by N, not N-1)
    features.burst_std = std::sqrt(burst_variance_sum / features.num_processes);
    features.priority_std = std::sqrt(priority_variance_sum / features.num_processes);
    
    features.short_job_ratio = static_cast<double>(short_jobs) / features.num_processes;
    features.long_job_ratio = static_cast<double>(long_jobs) / features.num_processes;
    
    int arrival_range = max_arrival - min_arrival + 1;
    if (arrival_range > 0) {
        features.arrival_rate = static_cast<double>(features.num_processes) / arrival_range;
    } else {
        // Fallback for extreme edge cases where max_arrival < min_arrival is possible (shouldn't happen)
        features.arrival_rate = static_cast<double>(features.num_processes);
    }
    
    return features;
}
