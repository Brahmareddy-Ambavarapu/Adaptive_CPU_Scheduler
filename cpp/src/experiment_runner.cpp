#include "experiment_runner.hpp"
#include "workload_generator.hpp"
#include "rule_based_adaptive_scheduler.hpp"
#include "ml_adaptive_scheduler.hpp"
#include "simulator.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, 0777)
#endif

void createDirectories(const std::string& base_dir) {
    MKDIR(base_dir.c_str());
    MKDIR((base_dir + "experiments/").c_str());
    MKDIR((base_dir + "summaries/").c_str());
}

double getMedian(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size() / 2;
    if (v.size() % 2 == 0) {
        return (v[n - 1] + v[n]) / 2.0;
    }
    return v[n];
}

void ExperimentRunner::run(const ExperimentConfig& config) {
    createDirectories(config.output_dir);
    
    std::string csv_path = config.output_dir + "experiments/per_workload.csv";
    std::ofstream out(csv_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << csv_path << "\n";
        return;
    }
    
    // CSV Header
    out << "workload_id,oracle_scheduler,oracle_optimal_score,"
        << "fcfs_score,sjf_score,rr_score,priority_score,"
        << "rule_based_scheduler,rule_based_score,rule_based_regret_percent,"
        << "ml_predicted_scheduler,ml_prediction_confidence,ml_score,ml_regret_percent,"
        << "fcfs_waiting,fcfs_response,fcfs_turnaround,fcfs_context_switches,"
        << "sjf_waiting,sjf_response,sjf_turnaround,sjf_context_switches,"
        << "rr_waiting,rr_response,rr_turnaround,rr_context_switches,"
        << "priority_waiting,priority_response,priority_turnaround,priority_context_switches\n";

    WorkloadGenerator generator(config.random_seed);
    RuleBasedConfig rb_config;
    rb_config.round_robin_quantum = config.rr_quantum;
    RuleBasedAdaptiveScheduler rule_based_scheduler(rb_config);
    MLAdaptiveScheduler ml_scheduler(config.rr_quantum);
    
    AdaptiveStats ml_stats;
    AdaptiveStats rb_stats;
    
    std::vector<double> ml_regrets;
    std::vector<double> rb_regrets;
    std::vector<double> ml_scores;
    std::vector<double> rb_scores;
    
    std::cout << "Starting Experiment with " << config.num_workloads << " workloads...\n";
    
    for (int i = 0; i < config.num_workloads; ++i) {
        // Generate clean workload
        WorkloadConfig wl_config;
        wl_config.num_processes = 5 + (i % 16); // 5 to 20 processes varying deterministically
        wl_config.seed = config.random_seed + i;
        auto workload = generator.generate(WorkloadType::Mixed, wl_config);
        
        if (workload.empty()) continue;

        // 1. Evaluate Oracle
        OracleDecision oracle = OracleScheduler::evaluate(workload, config.rr_quantum);
        
        // 2. Evaluate Rule-Based
        std::vector<Process> rb_workload = workload;
        Simulator sim_rb;
        AdaptiveDecision rb_decision = rule_based_scheduler.selectScheduler(rb_workload);
        std::string rb_name;
        switch(rb_decision.selected_scheduler) {
            case SchedulerType::FCFS: rb_name = "FCFS"; break;
            case SchedulerType::SJF: rb_name = "SJF"; break;
            case SchedulerType::RR: rb_name = "RR"; break;
            case SchedulerType::Priority: rb_name = "Priority"; break;
            default: rb_name = "Unknown";
        }
        rule_based_scheduler.run(sim_rb, rb_workload);
        SchedulerMetrics rb_metrics = Metrics::calculateMetrics(rb_name, sim_rb);
        
        // 3. Evaluate ML-Based
        std::vector<Process> ml_workload = workload;
        Simulator sim_ml;
        MLDecisionResult ml_decision = ml_scheduler.selectScheduler(ml_workload);
        ml_scheduler.run(sim_ml, ml_workload);
        SchedulerMetrics ml_metrics = Metrics::calculateMetrics(ml_decision.selected_scheduler_name, sim_ml);
        
        // Normalize adaptive metrics against Oracle's maximums!
        std::vector<SchedulerMetrics> adaptive_metrics = { rb_metrics, ml_metrics };
        Metrics::normalizeAndScoreWithReference(adaptive_metrics, oracle.maximums);
        rb_metrics = adaptive_metrics[0];
        ml_metrics = adaptive_metrics[1];
        
        // Calculate Regret safely
        auto calc_regret = [](double adaptive_score, double oracle_score) -> double {
            if (oracle_score <= 1e-9) {
                return (adaptive_score <= 1e-9) ? 0.0 : std::numeric_limits<double>::infinity();
            }
            return ((adaptive_score - oracle_score) / oracle_score) * 100.0;
        };
        
        double rb_regret = calc_regret(rb_metrics.objective_score, oracle.optimal_score);
        double ml_regret = calc_regret(ml_metrics.objective_score, oracle.optimal_score);
        
        // Collect Stats
        ml_regrets.push_back(ml_regret);
        rb_regrets.push_back(rb_regret);
        ml_scores.push_back(ml_metrics.objective_score);
        rb_scores.push_back(rb_metrics.objective_score);
        
        // Aggregation for Rule-Based
        rb_stats.average_score += rb_metrics.objective_score;
        rb_stats.average_waiting += rb_metrics.average_waiting_time;
        rb_stats.average_response += rb_metrics.average_response_time;
        rb_stats.average_turnaround += rb_metrics.average_turnaround_time;
        rb_stats.average_context_switches += rb_metrics.context_switches;
        if (rb_regret < rb_stats.min_regret) rb_stats.min_regret = rb_regret;
        if (rb_regret > rb_stats.worst_regret) rb_stats.worst_regret = rb_regret;
        if (rb_regret <= 1e-9) rb_stats.zero_regret_count++;
        if (rb_regret <= 5.0) rb_stats.within_5_percent_count++;
        if (rb_regret <= 10.0) rb_stats.within_10_percent_count++;
        
        // Aggregation for ML-Based
        ml_stats.average_score += ml_metrics.objective_score;
        ml_stats.average_waiting += ml_metrics.average_waiting_time;
        ml_stats.average_response += ml_metrics.average_response_time;
        ml_stats.average_turnaround += ml_metrics.average_turnaround_time;
        ml_stats.average_context_switches += ml_metrics.context_switches;
        if (ml_regret < ml_stats.min_regret) ml_stats.min_regret = ml_regret;
        if (ml_regret > ml_stats.worst_regret) ml_stats.worst_regret = ml_regret;
        if (ml_regret <= 1e-9) ml_stats.zero_regret_count++;
        if (ml_regret <= 5.0) ml_stats.within_5_percent_count++;
        if (ml_regret <= 10.0) ml_stats.within_10_percent_count++;
        
        if (ml_decision.selected_scheduler_name == oracle.optimal_scheduler_name) {
            ml_stats.correct_predictions++;
        }

        // CSV Row
        out << i << "," << oracle.optimal_scheduler_name << "," << oracle.optimal_score << ",";
        for (int j = 0; j < 4; ++j) out << oracle.base_metrics[j].objective_score << ",";
        out << rb_name << "," << rb_metrics.objective_score << "," << rb_regret << ",";
        out << ml_decision.selected_scheduler_name << "," << ml_decision.confidence << "," << ml_metrics.objective_score << "," << ml_regret << ",";
        for (int j = 0; j < 4; ++j) {
            out << oracle.base_metrics[j].average_waiting_time << ","
                << oracle.base_metrics[j].average_response_time << ","
                << oracle.base_metrics[j].average_turnaround_time << ","
                << oracle.base_metrics[j].context_switches;
            if (j < 3) out << ",";
        }
        out << "\n";
        
        if ((i + 1) % 100 == 0) {
            std::cout << "Processed " << (i + 1) << " workloads...\n";
        }
    }
    
    out.close();
    
    // Process final stats
    auto finalize_stats = [&](AdaptiveStats& s, std::vector<double>& regrets, std::vector<double>& scores) {
        s.average_score /= config.num_workloads;
        s.average_waiting /= config.num_workloads;
        s.average_response /= config.num_workloads;
        s.average_turnaround /= config.num_workloads;
        s.average_context_switches /= config.num_workloads;
        
        double sum_reg = 0;
        for (double r : regrets) sum_reg += r;
        s.average_regret = sum_reg / config.num_workloads;
        
        s.median_score = getMedian(scores);
        s.median_regret = getMedian(regrets);
    };
    
    finalize_stats(ml_stats, ml_regrets, ml_scores);
    finalize_stats(rb_stats, rb_regrets, rb_scores);
    
    // Write summaries
    std::string summary_path = config.output_dir + "summaries/performance_summary.csv";
    std::ofstream sum_out(summary_path);
    sum_out << "scheduler,average_score,median_score,average_waiting,average_response,average_turnaround,average_context_switches,average_regret,median_regret,worst_regret,zero_regret_percent,within_5_percent,within_10_percent\n";
    
    auto write_sum = [&](const std::string& name, const AdaptiveStats& s) {
        sum_out << name << "," << s.average_score << "," << s.median_score << "," 
                << s.average_waiting << "," << s.average_response << "," << s.average_turnaround << "," 
                << s.average_context_switches << "," << s.average_regret << "," << s.median_regret << "," 
                << s.worst_regret << "," << (s.zero_regret_count * 100.0 / config.num_workloads) << ","
                << (s.within_5_percent_count * 100.0 / config.num_workloads) << ","
                << (s.within_10_percent_count * 100.0 / config.num_workloads) << "\n";
    };
    write_sum("Rule-Based Adaptive", rb_stats);
    write_sum("ML-Based Adaptive", ml_stats);
    sum_out.close();
    
    std::string class_path = config.output_dir + "summaries/ml_selection_summary.csv";
    std::ofstream class_out(class_path);
    class_out << "total_workloads,ml_accuracy_percent\n";
    class_out << config.num_workloads << "," << (ml_stats.correct_predictions * 100.0 / config.num_workloads) << "\n";
    class_out.close();

    std::cout << "\nExperiment Complete!\n";
    std::cout << "Data written to " << config.output_dir << "\n";
}
