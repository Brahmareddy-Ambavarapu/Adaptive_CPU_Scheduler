#pragma once
#include "simulator.hpp"

class Scheduler {
public:
    virtual ~Scheduler() = default;

    // Core method that implements the scheduling algorithm
    virtual void run(Simulator& simulator) = 0;
};
