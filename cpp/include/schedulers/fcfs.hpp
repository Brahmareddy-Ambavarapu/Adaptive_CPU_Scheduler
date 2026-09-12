#pragma once
#include "scheduler.hpp"

class FCFSScheduler : public Scheduler {
public:
    void run(Simulator& simulator) override;
};
