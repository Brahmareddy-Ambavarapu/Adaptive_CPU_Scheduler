#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include "dataset_generator.hpp"
#include "workload_generator.hpp"
#include "metrics.hpp"

using namespace std;

bool approx_equal(double a, double b, double epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

// TEST 1
void test_feature_extraction() {
    DatasetGenerator gen;
    std::vector<Process> workload = {
        Process(1, 0, 5, 2),
        Process(2, 2, 3, 1),
        Process(3, 4, 10, 3)
    };
    DatasetRow row;
    gen.extractFeatures(row, workload);
    
    assert(row.num_processes == 3);
    assert(approx_equal(row.average_burst, 6.0));
    assert(approx_equal(row.min_burst, 3));
    assert(approx_equal(row.max_burst, 10));
    
    // std calculation: mean=6
    // (5-6)^2=1, (3-6)^2=9, (10-6)^2=16 -> sum = 26.
    // var = 26/3
    assert(approx_equal(row.burst_std, std::sqrt(26.0/3.0)));
    
    // short jobs: burst <= 6 (processes 1 and 2) -> 2/3
    assert(approx_equal(row.short_job_ratio, 2.0/3.0));
    // long jobs: burst > 6 (process 3) -> 1/3
    assert(approx_equal(row.long_job_ratio, 1.0/3.0));
}

// TEST 2
void test_arrival_rate() {
    DatasetGenerator gen;
    std::vector<Process> workload = {
        Process(1, 10, 5, 2),
        Process(2, 12, 3, 1),
        Process(3, 14, 10, 3)
    };
    DatasetRow row;
    gen.extractFeatures(row, workload);
    
    // min_arr=10, max_arr=14. time_span = 14 - 10 + 1 = 5
    // arrival_rate = 3 / 5 = 0.6
    assert(approx_equal(row.arrival_rate, 0.6));
}

// TEST 3, 4, 5, 6
void test_evaluations() {
    DatasetGenerator gen;
    std::vector<Process> workload = {
        Process(1, 0, 10, 1),
        Process(2, 1, 1, 1),
        Process(3, 2, 1, 1)
    };
    std::string target = gen.evaluateSchedulers(workload, 4);
    assert(target == "FCFS" || target == "SJF" || target == "RR" || target == "Priority");
    
    // State isolation: if we re-evaluate, the workload is unchanged (const ref)
    assert(workload[0].burst_time == 10); 
}

// TEST 7
void test_dataset_generation() {
    DatasetGenerator gen;
    DatasetConfig config;
    config.number_of_workloads = 20;
    
    auto ds = gen.generateDataset(config);
    assert(ds.size() == 20);
}

// TEST 8 & 9
void test_determinism() {
    DatasetGenerator gen;
    DatasetConfig config;
    config.number_of_workloads = 5;
    config.seed = 42;
    
    auto ds1 = gen.generateDataset(config);
    auto ds2 = gen.generateDataset(config);
    
    for (size_t i = 0; i < ds1.size(); ++i) {
        assert(approx_equal(ds1[i].average_burst, ds2[i].average_burst));
        assert(ds1[i].target_scheduler == ds2[i].target_scheduler);
    }
    
    config.seed = 999;
    auto ds3 = gen.generateDataset(config);
    bool diff = false;
    for (size_t i = 0; i < ds1.size(); ++i) {
        if (!approx_equal(ds1[i].average_burst, ds3[i].average_burst)) {
            diff = true;
            break;
        }
    }
    assert(diff);
}

// TEST 10, 11
void test_csv() {
    DatasetGenerator gen;
    DatasetConfig config;
    config.number_of_workloads = 5;
    config.output_path = "test_data.csv";
    auto ds = gen.generateDataset(config);
    gen.writeCsv(ds, config.output_path);
    
    std::ifstream in(config.output_path);
    assert(in.is_open());
    std::string header;
    std::getline(in, header);
    assert(header.find("workload_id") != std::string::npos);
    assert(header.find("target_scheduler") != std::string::npos);
    
    int row_count = 0;
    std::string line;
    while(std::getline(in, line)) {
        if(!line.empty()) row_count++;
    }
    assert(row_count == 5);
}

// TEST 12, 13, 14, 15
void test_features_validity() {
    DatasetGenerator gen;
    DatasetConfig config;
    config.number_of_workloads = 100;
    auto ds = gen.generateDataset(config);
    
    for (const auto& row : ds) {
        assert(std::isfinite(row.num_processes));
        assert(std::isfinite(row.average_burst));
        assert(std::isfinite(row.burst_std));
        assert(std::isfinite(row.arrival_rate));
        assert(std::isfinite(row.short_job_ratio));
        assert(std::isfinite(row.long_job_ratio));
        
        assert(row.short_job_ratio >= 0.0 && row.short_job_ratio <= 1.0);
        assert(row.long_job_ratio >= 0.0 && row.long_job_ratio <= 1.0);
        assert(approx_equal(row.short_job_ratio + row.long_job_ratio, 1.0));
        
        assert(row.target_scheduler == "FCFS" || 
               row.target_scheduler == "SJF" || 
               row.target_scheduler == "RR" || 
               row.target_scheduler == "Priority");
    }
}

int main() {
    test_feature_extraction();
    test_arrival_rate();
    test_evaluations();
    test_dataset_generation();
    test_determinism();
    test_csv();
    test_features_validity();
    cout << "All dataset generator tests passed!\n";
    return 0;
}
