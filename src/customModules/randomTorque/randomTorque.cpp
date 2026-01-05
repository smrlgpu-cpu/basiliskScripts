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
    this->controlMode = MODE_UNIFORM;
    
    
    v3SetZero(this->currentFinalTorque);

    // [Multisine 설정]
    // Nyquist 0.5Hz (1.0s step) -> Max Freq 0.4Hz
    this->numSineComponents = 10;
    double minFreq = 0.01;
    double maxFreq = 0.4; 
    
    this->sineFreqs.resize(this->numSineComponents);
    for(int i=0; i<this->numSineComponents; ++i) {
        double exponent = std::log10(minFreq) + (std::log10(maxFreq) - std::log10(minFreq)) * ((double)i / (double)(this->numSineComponents - 1));
        this->sineFreqs[i] = std::pow(10.0, exponent);
    }
    this->sinePhases.resize(3, std::vector<double>(this->numSineComponents));
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

    // Multisine 위상 랜덤화
    for(int axis=0; axis<3; ++axis) {
        for(int k=0; k<this->numSineComponents; ++k) {
            this->sinePhases[axis][k] = this->dist(this->rng) * 2.0 * M_PI;
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
        double timeSec = CurrentSimNanos * 1.0e-9;

        for(int i=0; i<3; i++) {
            double val = 0.0;
            
            if (this->controlMode == MODE_MULTISINE) { // Mode 1
                for(int k=0; k<this->numSineComponents; ++k) {
                    double freq = this->sineFreqs[k];
                    double phase = this->sinePhases[i][k];
                    val += std::sin(2.0 * M_PI * freq * timeSec + phase);
                }
                val = (val / (double)this->numSineComponents) * this->torqueMagnitude * 2.0;
                
                if(val > this->torqueMagnitude) val = this->torqueMagnitude;
                if(val < -this->torqueMagnitude) val = -this->torqueMagnitude;
                
            } else if (this->controlMode == MODE_APRBS) { // Mode 2
                double randVal = this->dist(this->rng);
                val = ((randVal * 2.0) - 1.0) * this->torqueMagnitude;
                
            } else { // MODE_UNIFORM (Mode 0)
                double randVal = this->dist(this->rng);
                val = ((randVal * 2.0) - 1.0) * this->torqueMagnitude;
            }

            this->currentFinalTorque[i] = val;
        }

        if (this->controlMode == MODE_APRBS) {
            std::uniform_int_distribution<int> stepDist(1, 4);
            int randomSteps = stepDist(this->rng); 
            uint64_t variableHold = this->holdPeriodNs * randomSteps;
            if (variableHold < this->holdPeriodNs) variableHold = this->holdPeriodNs;
            this->nextUpdateNs = CurrentSimNanos + variableHold;
        } 
        else {
            uint64_t period = (this->holdPeriodNs > 0) ? this->holdPeriodNs : 100000000; 
            this->nextUpdateNs = CurrentSimNanos + period;
        }
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


void RandomTorque::setControlMode(int mode) {
    this->controlMode = mode;
}