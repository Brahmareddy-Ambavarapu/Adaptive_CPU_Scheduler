#pragma once

#include "process.hpp"
#include "workload_generator.hpp"
#include <string>
#include <vector>

struct DatasetRow {
    int workload_id;

    double num_processes;
    double average_burst;
    double burst_std;
    double min_burst;
    double max_burst;

    double arrival_rate;

    double average_priority;
    double priority_std;

    double short_job_ratio;
    double long_job_ratio;

    std::string target_scheduler;
};

struct DatasetConfig {
    int number_of_workloads = 1000;
    unsigned int seed = 42;
    int round_robin_quantum = 4;
    std::string output_path = "data/cpu_scheduling_dataset.csv";
    WorkloadConfig workload_config;
};

class DatasetGenerator {
public:
    DatasetGenerator();

    // Generates a dataset in memory
    std::vector<DatasetRow> generateDataset(const DatasetConfig& config);

    // Writes the dataset to a CSV file
    void writeCsv(const std::vector<DatasetRow>& dataset, const std::string& path);

public:
    DatasetRow evaluateWorkload(int id, const std::vector<Process>& workload, int quantum);
    void extractFeatures(DatasetRow& row, const std::vector<Process>& workload);
    std::string evaluateSchedulers(const std::vector<Process>& workload, int quantum);
};
