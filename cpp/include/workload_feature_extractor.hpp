#ifndef WORKLOAD_FEATURE_EXTRACTOR_HPP
#define WORKLOAD_FEATURE_EXTRACTOR_HPP

#include "process.hpp"
#include "workload_features.hpp"
#include <vector>

class WorkloadFeatureExtractor {
public:
    static WorkloadFeatures extract(const std::vector<Process>& processes);
};

#endif // WORKLOAD_FEATURE_EXTRACTOR_HPP
