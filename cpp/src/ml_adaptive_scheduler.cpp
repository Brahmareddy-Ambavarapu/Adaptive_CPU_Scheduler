#include "ml_adaptive_scheduler.hpp"
#include "workload_feature_extractor.hpp"
#include "schedulers/fcfs.hpp"
#include "schedulers/sjf.hpp"
#include "schedulers/rr.hpp"
#include "schedulers/priority.hpp"
#include <iostream>
#include <stdexcept>
#include <memory>
#include <sstream>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <array>
#include <regex>
#include <chrono>

MLAdaptiveScheduler::MLAdaptiveScheduler(int round_robin_quantum) 
    : round_robin_quantum(round_robin_quantum) {
    if (round_robin_quantum <= 0) {
        throw std::invalid_argument("Round Robin quantum must be positive.");
    }
}

std::string MLAdaptiveScheduler::runPythonInference(const std::array<double, 10>& features) const {
    static int counter = 0;
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::string temp_file = "ml_inference_temp_" + std::to_string(now) + "_" + std::to_string(rand() % 1000000) + "_" + std::to_string(++counter) + ".json";
    std::ostringstream cmd;
    cmd << "python ml/predict_scheduler.py";
    for (double f : features) {
        cmd << " " << f;
    }
    cmd << " > " << temp_file;
    
    // Execute command synchronously
    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        // Continue anyway to try and read the error file if it was created
    }
    
    // Read the file
    std::ifstream infile(temp_file);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to read Python inference output file.");
    }
    
    std::string result;
    std::string line;
    while (std::getline(infile, line)) {
        result += line;
    }
    infile.close();
    
    // Delete temp file
    std::remove(temp_file.c_str());
    
    return result;
}

MLDecisionResult MLAdaptiveScheduler::selectScheduler(const std::vector<Process>& processes) const {
    MLDecisionResult decision;
    
    if (processes.empty()) {
        throw std::invalid_argument("Cannot perform ML inference on an empty workload.");
    }
    
    // 1. Extract purely pre-execution workload characteristics
    WorkloadFeatures feats = WorkloadFeatureExtractor::extract(processes);
    
    // 2. Convert features into exact ordering for Step 11 model
    decision.features = {
        static_cast<double>(feats.num_processes),
        feats.average_burst,
        feats.burst_std,
        static_cast<double>(feats.min_burst),
        static_cast<double>(feats.max_burst),
        feats.arrival_rate,
        feats.average_priority,
        feats.priority_std,
        feats.short_job_ratio,
        feats.long_job_ratio
    };
    
    // 3. Call python CLI for inference
    std::string output = runPythonInference(decision.features);
    
    // 4. Very simple JSON extraction (without adding a full JSON parsing library to C++)
    // We expect {"predicted_label": 1, "scheduler": "SJF", "confidence": 0.87} or an error.
    
    if (output.find("\"error\"") != std::string::npos) {
        throw std::runtime_error("ML Inference failed: " + output);
    }
    
    std::regex label_regex("\"predicted_label\"\\s*:\\s*(\\d+)");
    std::regex scheduler_regex("\"scheduler\"\\s*:\\s*\"([A-Za-z]+)\"");
    std::regex confidence_regex("\"confidence\"\\s*:\\s*([0-9\\.-]+)");
    
    std::smatch label_match, scheduler_match, confidence_match;
    
    if (std::regex_search(output, label_match, label_regex) && label_match.size() > 1) {
        decision.predicted_label = std::stoi(label_match.str(1));
    } else {
        throw std::runtime_error("Failed to parse 'predicted_label' from inference output.");
    }
    
    if (std::regex_search(output, scheduler_match, scheduler_regex) && scheduler_match.size() > 1) {
        decision.selected_scheduler_name = scheduler_match.str(1);
    } else {
        throw std::runtime_error("Failed to parse 'scheduler' from inference output.");
    }
    
    if (std::regex_search(output, confidence_match, confidence_regex) && confidence_match.size() > 1) {
        decision.confidence = std::stod(confidence_match.str(1));
    } else {
        decision.confidence = -1.0;
    }
    
    // Map string/label back to SchedulerType explicitly
    if (decision.selected_scheduler_name == "FCFS" || decision.predicted_label == 0) {
        decision.selected_scheduler = SchedulerType::FCFS;
    } else if (decision.selected_scheduler_name == "SJF" || decision.predicted_label == 1) {
        decision.selected_scheduler = SchedulerType::SJF;
    } else if (decision.selected_scheduler_name == "RR" || decision.predicted_label == 2) {
        decision.selected_scheduler = SchedulerType::RR;
    } else if (decision.selected_scheduler_name == "Priority" || decision.predicted_label == 3) {
        decision.selected_scheduler = SchedulerType::Priority;
    } else {
        throw std::runtime_error("Unrecognized scheduler predicted: " + decision.selected_scheduler_name);
    }
    
    return decision;
}

void MLAdaptiveScheduler::run(Simulator& simulator, const std::vector<Process>& processes) const {
    MLDecisionResult decision = selectScheduler(processes);
    
    // Exactly one scheduler is invoked.
    std::unique_ptr<Scheduler> scheduler;
    
    switch (decision.selected_scheduler) {
        case SchedulerType::FCFS:
            scheduler = std::make_unique<FCFSScheduler>();
            break;
        case SchedulerType::SJF:
            scheduler = std::make_unique<SJFScheduler>();
            break;
        case SchedulerType::RR:
            scheduler = std::make_unique<RRScheduler>(round_robin_quantum);
            break;
        case SchedulerType::Priority:
            scheduler = std::make_unique<PriorityScheduler>();
            break;
        default:
            throw std::runtime_error("Invalid SchedulerType resolved.");
    }
    
    simulator.reset();
    for (const auto& p : processes) {
        simulator.addProcess(p);
    }
    
    scheduler->run(simulator);
}
