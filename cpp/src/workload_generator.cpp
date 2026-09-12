#include "workload_generator.hpp"
#include <stdexcept>
#include <algorithm>

WorkloadGenerator::WorkloadGenerator(unsigned int seed) : gen(seed) {}

std::vector<Process> WorkloadGenerator::generate(WorkloadType type, const WorkloadConfig& config) {
    if (config.num_processes <= 0) throw std::invalid_argument("num_processes must be > 0");
    if (config.min_burst_time <= 0) throw std::invalid_argument("min_burst_time must be > 0");
    if (config.max_burst_time < config.min_burst_time) throw std::invalid_argument("max_burst_time < min_burst_time");
    if (config.min_priority <= 0) throw std::invalid_argument("min_priority must be > 0");
    if (config.max_priority < config.min_priority) throw std::invalid_argument("max_priority < min_priority");
    if (config.max_arrival_time < 0) throw std::invalid_argument("max_arrival_time must be >= 0");
    if (config.round_robin_quantum <= 0) throw std::invalid_argument("quantum must be > 0");

    std::vector<Process> processes;
    processes.reserve(config.num_processes);

    std::uniform_int_distribution<> prio_dist(config.min_priority, config.max_priority);
    std::uniform_int_distribution<> default_arrival(0, config.max_arrival_time);
    
    // Arrival distributions
    std::uniform_int_distribution<> high_arrival_dist(0, std::max(1, config.max_arrival_time / 10));
    std::uniform_int_distribution<> low_arrival_dist(0, config.max_arrival_time * 5);

    // Burst distributions
    std::uniform_int_distribution<> mixed_burst(config.min_burst_time, config.max_burst_time);
    std::uniform_int_distribution<> short_burst(config.min_burst_time, std::max(config.min_burst_time, config.max_burst_time / 4));
    std::uniform_int_distribution<> long_burst(std::max(config.min_burst_time, config.max_burst_time / 2), config.max_burst_time);
    std::uniform_int_distribution<> cpu_heavy_burst(config.max_burst_time, config.max_burst_time * 5);
    
    std::bernoulli_distribution bimodal_coin(0.5);

    for (int i = 1; i <= config.num_processes; ++i) {
        int arrival = 0;
        int burst = 1;
        int priority = prio_dist(gen);

        // Arrival generation
        if (type == WorkloadType::HighArrivalRate) {
            arrival = high_arrival_dist(gen);
        } else if (type == WorkloadType::LowArrivalRate) {
            arrival = low_arrival_dist(gen);
        } else {
            arrival = default_arrival(gen);
        }

        // Burst generation
        if (type == WorkloadType::ShortJobHeavy) {
            burst = short_burst(gen);
        } else if (type == WorkloadType::LongJobHeavy) {
            burst = long_burst(gen);
        } else if (type == WorkloadType::CpuHeavy) {
            burst = cpu_heavy_burst(gen);
        } else if (type == WorkloadType::Bimodal) {
            burst = bimodal_coin(gen) ? short_burst(gen) : long_burst(gen);
        } else {
            burst = mixed_burst(gen);
        }

        // Ensure PriorityHeavy actually forces wide spread
        if (type == WorkloadType::PriorityHeavy) {
            std::uniform_int_distribution<> wide_prio(1, 100);
            priority = wide_prio(gen);
        }

        processes.emplace_back(i, arrival, burst, priority);
    }

    return processes;
}
