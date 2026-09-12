#include "dataset_generator.hpp"
#include "simulator.hpp"
#include "schedulers/fcfs.hpp"
#include "schedulers/sjf.hpp"
#include "schedulers/rr.hpp"
#include "schedulers/priority.hpp"
#include "metrics.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <iostream>

DatasetGenerator::DatasetGenerator() {}

std::vector<DatasetRow> DatasetGenerator::generateDataset(const DatasetConfig& config) {
    std::vector<DatasetRow> dataset;
    dataset.reserve(config.number_of_workloads);

    std::vector<WorkloadType> families = {
        WorkloadType::ShortJobHeavy,
        WorkloadType::LongJobHeavy,
        WorkloadType::Mixed,
        WorkloadType::CpuHeavy,
        WorkloadType::HighArrivalRate,
        WorkloadType::LowArrivalRate,
        WorkloadType::PriorityHeavy,
        WorkloadType::Bimodal
    };

    for (int i = 0; i < config.number_of_workloads; ++i) {
        WorkloadType type = families[i % families.size()];
        unsigned int w_seed = config.seed + i;
        WorkloadGenerator gen(w_seed);
        
        auto workload = gen.generate(type, config.workload_config);
        
        DatasetRow row = evaluateWorkload(i + 1, workload, config.round_robin_quantum);
        dataset.push_back(row);
    }
    return dataset;
}

DatasetRow DatasetGenerator::evaluateWorkload(int id, const std::vector<Process>& workload, int quantum) {
    DatasetRow row;
    row.workload_id = id;
    extractFeatures(row, workload);
    row.target_scheduler = evaluateSchedulers(workload, quantum);
    return row;
}

void DatasetGenerator::extractFeatures(DatasetRow& row, const std::vector<Process>& workload) {
    row.num_processes = workload.size();
    if (workload.empty()) return;

    double total_burst = 0;
    double total_prio = 0;
    double min_burst = workload[0].burst_time;
    double max_burst = workload[0].burst_time;
    double min_arrival = workload[0].arrival_time;
    double max_arrival = workload[0].arrival_time;

    for (const auto& p : workload) {
        total_burst += p.burst_time;
        total_prio += p.priority;
        if (p.burst_time < min_burst) min_burst = p.burst_time;
        if (p.burst_time > max_burst) max_burst = p.burst_time;
        if (p.arrival_time < min_arrival) min_arrival = p.arrival_time;
        if (p.arrival_time > max_arrival) max_arrival = p.arrival_time;
    }

    row.average_burst = total_burst / row.num_processes;
    row.average_priority = total_prio / row.num_processes;
    row.min_burst = min_burst;
    row.max_burst = max_burst;

    double burst_var_sum = 0;
    double prio_var_sum = 0;
    int short_count = 0;
    int long_count = 0;

    for (const auto& p : workload) {
        burst_var_sum += std::pow(p.burst_time - row.average_burst, 2);
        prio_var_sum += std::pow(p.priority - row.average_priority, 2);
        
        // threshold definition: <= avg is short, > avg is long
        if (p.burst_time <= row.average_burst) short_count++;
        if (p.burst_time > row.average_burst) long_count++;
    }

    // Population standard deviation
    row.burst_std = std::sqrt(burst_var_sum / row.num_processes);
    row.priority_std = std::sqrt(prio_var_sum / row.num_processes);
    
    row.short_job_ratio = static_cast<double>(short_count) / row.num_processes;
    row.long_job_ratio = static_cast<double>(long_count) / row.num_processes;

    double time_span = max_arrival - min_arrival + 1.0;
    row.arrival_rate = row.num_processes / time_span;
}

std::string DatasetGenerator::evaluateSchedulers(const std::vector<Process>& workload, int quantum) {
    std::vector<SchedulerMetrics> results;

    // FCFS
    Simulator sim_fcfs;
    for (const auto& p : workload) sim_fcfs.addProcess(p);
    FCFSScheduler fcfs;
    fcfs.run(sim_fcfs);
    results.push_back(Metrics::calculateMetrics("FCFS", sim_fcfs));

    // SJF
    Simulator sim_sjf;
    for (const auto& p : workload) sim_sjf.addProcess(p);
    SJFScheduler sjf;
    sjf.run(sim_sjf);
    results.push_back(Metrics::calculateMetrics("SJF", sim_sjf));

    // RR
    Simulator sim_rr;
    for (const auto& p : workload) sim_rr.addProcess(p);
    RRScheduler rr(quantum);
    rr.run(sim_rr);
    results.push_back(Metrics::calculateMetrics("RR", sim_rr));

    // Priority
    Simulator sim_pri;
    for (const auto& p : workload) sim_pri.addProcess(p);
    PriorityScheduler pri;
    pri.run(sim_pri);
    results.push_back(Metrics::calculateMetrics("Priority", sim_pri));

    Metrics::normalizeAndScore(results);

    // Tie-breaking: FCFS > SJF > RR > Priority
    double min_score = results[0].objective_score;
    std::string best_scheduler = results[0].scheduler_name;

    for (size_t i = 1; i < results.size(); ++i) {
        // Strict less-than guarantees the tie-breaking order (earlier index wins ties)
        if (results[i].objective_score < min_score) {
            min_score = results[i].objective_score;
            best_scheduler = results[i].scheduler_name;
        }
    }

    return best_scheduler;
}

void DatasetGenerator::writeCsv(const std::vector<DatasetRow>& dataset, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("Cannot open CSV file for writing");

    out << "workload_id,num_processes,average_burst,burst_std,min_burst,max_burst,"
        << "arrival_rate,average_priority,priority_std,short_job_ratio,long_job_ratio,target_scheduler\n";

    for (const auto& row : dataset) {
        out << row.workload_id << ","
            << row.num_processes << ","
            << row.average_burst << ","
            << row.burst_std << ","
            << row.min_burst << ","
            << row.max_burst << ","
            << row.arrival_rate << ","
            << row.average_priority << ","
            << row.priority_std << ","
            << row.short_job_ratio << ","
            << row.long_job_ratio << ","
            << row.target_scheduler << "\n";
    }
}
