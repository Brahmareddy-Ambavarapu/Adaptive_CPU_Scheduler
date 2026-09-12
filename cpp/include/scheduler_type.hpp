#ifndef SCHEDULER_TYPE_HPP
#define SCHEDULER_TYPE_HPP

#include <string>

enum class SchedulerType {
    FCFS,
    SJF,
    RR,
    Priority
};

inline std::string schedulerTypeToString(SchedulerType type) {
    switch (type) {
        case SchedulerType::FCFS:     return "FCFS";
        case SchedulerType::SJF:      return "SJF";
        case SchedulerType::RR:       return "RR";
        case SchedulerType::Priority: return "Priority";
        default:                      return "Unknown";
    }
}

#endif // SCHEDULER_TYPE_HPP
