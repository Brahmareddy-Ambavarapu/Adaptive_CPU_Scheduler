#pragma once
#include "scheduler.hpp"

class PriorityScheduler : public Scheduler {
public:
    void run(Simulator& simulator) override;
};
