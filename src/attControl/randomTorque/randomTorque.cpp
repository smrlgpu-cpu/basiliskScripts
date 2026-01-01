#include "randomTorque.h"
#include <iostream>
#include <chrono>

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
    this->ditherStd = 0.0;
    this->ditherDist = std::normal_distribution<double>(0.0, 1.0);
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

    // --- 1. Update Logic (Executed only when hold period expires) ---
    if (this->holdPeriodNs == 0 || CurrentSimNanos >= this->nextUpdateNs) {
        
        for(int i=0; i<3; i++) {
            double val = 0.0;
            double randVal = this->dist(this->rng); // [0.0, 1.0]

            if (this->controlMode == MODE_SATURATION) { 
                // [Mode 1: Saturation] 95% ~ 100% of Mag
                // jitter within the saturation boundary
                double sign = (randVal >= 0.5) ? 1.0 : -1.0;
                
                // New random draw for magnitude scaling
                double randMag = this->dist(this->rng); 
                double magScale = 0.95 + (randMag * 0.05); // [0.95, 1.00]
                
                val = sign * magScale * this->torqueMagnitude;
                
            } else if (this->controlMode == MODE_LOW) {
                // [Mode 2: Low Mag] -20% ~ +20%
                double normalized = (randVal * 2.0) - 1.0; // [-1.0, 1.0]
                val = normalized * 0.2 * this->torqueMagnitude;
                
            } else if (this->controlMode == MODE_MEDIUM) {
                // [Mode 3: Medium] 50% ~ 80%
                double sign = (randVal >= 0.5) ? 1.0 : -1.0;
                double randMag = this->dist(this->rng);
                double magScale = 0.5 + (randMag * 0.3); // [0.5, 0.8]
                val = sign * magScale * this->torqueMagnitude;

            } else {
                // [Mode 0: Uniform] -100% ~ +100%
                double normalized = (randVal * 2.0) - 1.0;
                val = normalized * this->torqueMagnitude;
            }
            
            // --- 2. Apply Dithering Here (Synchronized with Control Update) ---
            // Dithering is now part of the held value
            double noise = 0.0;
            if (this->ditherStd > 0.0) {
                noise = this->ditherDist(this->rng) * this->ditherStd;
            }
            
            // Store final combined torque
            this->currentFinalTorque[i] = val + noise;
        }

        // Schedule next update
        if (this->holdPeriodNs > 0) {
            this->nextUpdateNs = CurrentSimNanos + this->holdPeriodNs;
        }
    }

    // --- 3. Output the Held Torque ---
    // This value remains constant for 1.0s (including the dither noise)
    v3Copy(this->currentFinalTorque, outMsgBuffer.torqueRequestBody);

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

void RandomTorque::setDitherStd(double value) {
    if (value >= 0) {
        this->ditherStd = value;
    }
}

void RandomTorque::setControlMode(int mode) {
    this->controlMode = mode;
}