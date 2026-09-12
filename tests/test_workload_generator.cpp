#include <iostream>
#include <cassert>
#include <stdexcept>
#include <unordered_set>
#include "workload_generator.hpp"

using namespace std;

void test_size() {
    WorkloadConfig config;
    config.num_processes = 100;
    WorkloadGenerator gen(42);
    auto p = gen.generate(WorkloadType::Mixed, config);
    assert(p.size() == 100);
}

void test_unique_pids() {
    WorkloadConfig config;
    config.num_processes = 50;
    WorkloadGenerator gen(42);
    auto p = gen.generate(WorkloadType::Mixed, config);
    unordered_set<int> pids;
    for (int i = 0; i < 50; ++i) {
        assert(p[i].pid == i + 1);
        pids.insert(p[i].pid);
    }
    assert(pids.size() == 50);
}

void test_positive_burst_times() {
    WorkloadConfig config;
    WorkloadGenerator gen(42);
    auto p = gen.generate(WorkloadType::ShortJobHeavy, config);
    for (const auto& proc : p) assert(proc.burst_time > 0);
}

void test_valid_arrivals_and_priorities() {
    WorkloadConfig config;
    config.min_priority = 2;
    config.max_priority = 8;
    WorkloadGenerator gen(42);
    auto p = gen.generate(WorkloadType::Mixed, config);
    for (const auto& proc : p) {
        assert(proc.arrival_time >= 0);
        assert(proc.priority >= 2 && proc.priority <= 8);
    }
}

void test_deterministic_seed() {
    WorkloadConfig config;
    WorkloadGenerator gen1(123);
    auto p1 = gen1.generate(WorkloadType::Mixed, config);
    
    WorkloadGenerator gen2(123);
    auto p2 = gen2.generate(WorkloadType::Mixed, config);
    
    for (size_t i = 0; i < p1.size(); ++i) {
        assert(p1[i].pid == p2[i].pid);
        assert(p1[i].arrival_time == p2[i].arrival_time);
        assert(p1[i].burst_time == p2[i].burst_time);
        assert(p1[i].priority == p2[i].priority);
    }
}

void test_different_seeds() {
    WorkloadConfig config;
    WorkloadGenerator gen1(42);
    auto p1 = gen1.generate(WorkloadType::Mixed, config);
    
    WorkloadGenerator gen2(999);
    auto p2 = gen2.generate(WorkloadType::Mixed, config);
    
    bool diff = false;
    for (size_t i = 0; i < p1.size(); ++i) {
        if (p1[i].burst_time != p2[i].burst_time || p1[i].arrival_time != p2[i].arrival_time) {
            diff = true;
            break;
        }
    }
    assert(diff);
}

void test_invalid_config() {
    WorkloadGenerator gen(42);
    WorkloadConfig c1; c1.num_processes = 0;
    try { gen.generate(WorkloadType::Mixed, c1); assert(false); } catch(std::invalid_argument&) {}
    
    WorkloadConfig c2; c2.min_burst_time = 0;
    try { gen.generate(WorkloadType::Mixed, c2); assert(false); } catch(std::invalid_argument&) {}
}

void test_all_types() {
    WorkloadConfig config;
    WorkloadGenerator gen(42);
    std::vector<WorkloadType> types = {
        WorkloadType::ShortJobHeavy, WorkloadType::LongJobHeavy, WorkloadType::Mixed,
        WorkloadType::CpuHeavy, WorkloadType::HighArrivalRate, WorkloadType::LowArrivalRate,
        WorkloadType::PriorityHeavy, WorkloadType::Bimodal
    };
    for (auto type : types) {
        auto p = gen.generate(type, config);
        assert(!p.empty());
        for (const auto& proc : p) assert(proc.burst_time > 0);
    }
}

void test_short_vs_long_jobs() {
    WorkloadConfig config;
    config.num_processes = 200;
    config.min_burst_time = 1;
    config.max_burst_time = 100;
    
    WorkloadGenerator gen(42);
    auto short_jobs = gen.generate(WorkloadType::ShortJobHeavy, config);
    auto long_jobs = gen.generate(WorkloadType::LongJobHeavy, config);
    
    double avg_short = 0, avg_long = 0;
    for (const auto& p : short_jobs) avg_short += p.burst_time;
    for (const auto& p : long_jobs) avg_long += p.burst_time;
    
    avg_short /= short_jobs.size();
    avg_long /= long_jobs.size();
    
    assert(avg_short < avg_long);
    assert(avg_short <= (config.max_burst_time / 2.0));
    assert(avg_long >= (config.max_burst_time / 2.0));
}

void test_arrival_rates() {
    WorkloadConfig config;
    config.num_processes = 200;
    config.max_arrival_time = 100;
    
    WorkloadGenerator gen(42);
    auto high_arrival = gen.generate(WorkloadType::HighArrivalRate, config);
    auto low_arrival = gen.generate(WorkloadType::LowArrivalRate, config);
    
    double avg_high = 0, avg_low = 0;
    for (const auto& p : high_arrival) avg_high += p.arrival_time;
    for (const auto& p : low_arrival) avg_low += p.arrival_time;
    
    avg_high /= high_arrival.size();
    avg_low /= low_arrival.size();
    
    assert(avg_high < avg_low);
}

void test_bimodal() {
    WorkloadConfig config;
    config.num_processes = 200;
    config.min_burst_time = 1;
    config.max_burst_time = 100;
    
    WorkloadGenerator gen(42);
    auto bimodal = gen.generate(WorkloadType::Bimodal, config);
    
    int short_count = 0;
    int long_count = 0;
    int medium_count = 0;
    
    for (const auto& p : bimodal) {
        if (p.burst_time <= config.max_burst_time / 4) short_count++;
        else if (p.burst_time >= config.max_burst_time / 2) long_count++;
        else medium_count++;
    }
    
    assert(short_count > 0);
    assert(long_count > 0);
    assert(medium_count == 0);
}

int main() {
    test_size();
    test_unique_pids();
    test_positive_burst_times();
    test_valid_arrivals_and_priorities();
    test_deterministic_seed();
    test_different_seeds();
    test_invalid_config();
    test_all_types();
    test_short_vs_long_jobs();
    test_arrival_rates();
    test_bimodal();
    cout << "All workload generator tests passed!\n";
    return 0;
}
