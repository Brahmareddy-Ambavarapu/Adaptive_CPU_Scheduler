#pragma once
#include "scheduler.hpp"

class RRScheduler : public Scheduler {
private:
    int quantum;
public:
    explicit RRScheduler(int time_quantum);
    void run(Simulator& simulator) override;
};
