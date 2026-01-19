#include "randomTorque.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm> 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

RandomTorque::RandomTorque()
{
    if (this->seed == 0) {
        unsigned int timeSeed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        this->rng = std::mt19937(timeSeed);
    } else {
        this->rng = std::mt19937(this->seed);
    }
    
    this->dist = std::uniform_real_distribution<double>(0.0, 1.0);

    this->holdPeriodNs = 0;
    this->nextUpdateNs = 0;
    
    v3SetZero(this->currentFinalTorque);
}

RandomTorque::~RandomTorque()
{
    return;
}

void RandomTorque::Reset(uint64_t CurrentSimNanos)
{
    if (this->seed == 0) {
        unsigned int timeSeed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        this->rng = std::mt19937(timeSeed);
    } else {
        this->rng = std::mt19937(this->seed);
    }
    
    this->dist = std::uniform_real_distribution<double>(0.0, 1.0);

    if (this->vehConfigInMsg.isLinked()) {
        VehicleConfigMsgPayload vehConfigMsg = this->vehConfigInMsg();
        for (int i = 0; i < 9; i++) {
            this->ISCPntB_B[i] = vehConfigMsg.ISCPntB_B[i];
        }
    }

    this->nextUpdateNs = CurrentSimNanos; 
    v3SetZero(this->currentFinalTorque);

    CmdTorqueBodyMsgPayload outMsgBuffer = {};
    v3SetZero(outMsgBuffer.torqueRequestBody);
    this->cmdTorqueOutMsg.write(&outMsgBuffer, this->moduleID, CurrentSimNanos);
}

void RandomTorque::UpdateState(uint64_t CurrentSimNanos)
{
    CmdTorqueBodyMsgPayload outMsgBuffer;
    outMsgBuffer = this->cmdTorqueOutMsg.zeroMsgPayload;

    bool timeToUpdate = (CurrentSimNanos >= this->nextUpdateNs);

    if (timeToUpdate) {
        // Always use Uniform Random Logic
        for(int i=0; i<3; i++) {
            double randVal = this->dist(this->rng);
            double val = ((randVal * 2.0) - 1.0) * this->torqueMagnitude;
            this->currentFinalTorque[i] = val;
        }

        uint64_t period = (this->holdPeriodNs > 0) ? this->holdPeriodNs : 100000000; 
        this->nextUpdateNs = CurrentSimNanos + period;
    }
    
    // [출력]
    for(int i=0; i<3; i++) {
        outMsgBuffer.torqueRequestBody[i] = this->currentFinalTorque[i];
    }

    this->cmdTorqueOutMsg.write(&outMsgBuffer, this->moduleID, CurrentSimNanos);
}

void RandomTorque::setTorqueMagnitude(double value) {
    if (value >= 0.0) this->torqueMagnitude = value;
}

void RandomTorque::setSeed(unsigned int value) {
    this->seed = value;
    if (this->seed == 0) {
        unsigned int timeSeed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        this->rng = std::mt19937(timeSeed);
    } else {
        this->rng = std::mt19937(this->seed);
    }
}

void RandomTorque::setHoldPeriod(double seconds) {
    if (seconds >= 0) {
        this->holdPeriodNs = (uint64_t)(seconds * 1e9);
    }
}
