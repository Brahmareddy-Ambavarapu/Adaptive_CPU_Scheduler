#pragma once
#include "scheduler.hpp"

class SJFScheduler : public Scheduler {
public:
    void run(Simulator& simulator) override;
};
