#include "randomTorque.h"
#include <iostream>
#include <chrono>

/*! This is the constructor for the module class.  It sets default variable
    values and initializes the various parts of the model */
RandomTorque::RandomTorque()
{
    // Initialize random number generator with default seed
    if (this->seed == 0) {
        unsigned int timeSeed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        this->rng = std::mt19937(timeSeed);
    } else {
        this->rng = std::mt19937(this->seed);
    }
    
    // Initialize uniform distribution from -torqueMagnitude to +torqueMagnitude
    this->dist = std::uniform_real_distribution<double>(-this->torqueMagnitude, this->torqueMagnitude);

    // Initialize new parameters
    this->holdPeriodNs = 0;
    this->nextUpdateNs = 0;
    this->controlMode = MODE_UNIFORM;
    this->ditherStd = 0.0;
    this->ditherDist = std::normal_distribution<double>(0.0, 1.0);
    v3SetZero(this->currentBaseTorque);
}

/*! Module Destructor.  */
RandomTorque::~RandomTorque()
{
    return;
}


/*! This method is used to reset the module.

 */
void RandomTorque::Reset(uint64_t CurrentSimNanos)
{
    /*! - Reinitialize random number generator */
    if (this->seed == 0) {
        unsigned int timeSeed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        this->rng = std::mt19937(timeSeed);
    } else {
        this->rng = std::mt19937(this->seed);
    }
    
    // Reinitialize distribution with current torqueMagnitude
    this->dist = std::uniform_real_distribution<double>(-this->torqueMagnitude, this->torqueMagnitude);

    /*! - Read vehicle configuration message if linked */
    if (this->vehConfigInMsg.isLinked()) {
        VehicleConfigMsgPayload vehConfigMsg = this->vehConfigInMsg();
        // Copy inertia tensor
        for (int i = 0; i < 9; i++) {
            this->ISCPntB_B[i] = vehConfigMsg.ISCPntB_B[i];
        }
    }

    // Reset Hold Logic
    this->nextUpdateNs = CurrentSimNanos; 
    v3SetZero(this->currentBaseTorque);

    /* zero output message on reset */
    CmdTorqueBodyMsgPayload outMsgBuffer = {};       /*!< local output message copy */
    v3SetZero(outMsgBuffer.torqueRequestBody);
    this->cmdTorqueOutMsg.write(&outMsgBuffer, this->moduleID, CurrentSimNanos);
}


/*! This is the main method that gets called every time the module is updated.
    It generates a random torque vector and writes it to the output message.

 */
void RandomTorque::UpdateState(uint64_t CurrentSimNanos)
{
    CmdTorqueBodyMsgPayload outMsgBuffer;           /*!< local output message copy */
    AttGuidMsgPayload guidInMsgBuffer;               /*!< local copy of guidance input message */

    // always zero the output buffer first
    outMsgBuffer = this->cmdTorqueOutMsg.zeroMsgPayload;

    /*! - Read the optional input messages */
    if (this->guidInMsg.isLinked()) {
        guidInMsgBuffer = this->guidInMsg();
    }

    // --- 1. Base Torque Update (Hold Logic) ---
    // Update base torque if hold period is 0 (always update) or time threshold reached
    if (this->holdPeriodNs == 0 || CurrentSimNanos >= this->nextUpdateNs) {
        
        for(int i=0; i<3; i++) {
            double val = 0.0;
            
            if (this->controlMode == MODE_BANGBANG) { 
                // [Mode 1: Bang-Bang] -Max or +Max
                // Draw from uniform dist [-Mag, +Mag]. If >= 0, use +Mag, else -Mag.
                double r = this->dist(this->rng); 
                val = (r >= 0.0) ? this->torqueMagnitude : -this->torqueMagnitude;
                
            } else if (this->controlMode == MODE_LOW_MAG) {
                // [Mode 2: Low Mag] -0.1*Mag ~ +0.1*Mag
                // Scale the uniform distribution result by 0.1
                val = this->dist(this->rng) * 0.1;
                
            } else {
                // [Mode 0: Uniform] -Mag ~ +Mag
                val = this->dist(this->rng);
            }
            
            this->currentBaseTorque[i] = val;
        }

        // Schedule next update
        if (this->holdPeriodNs > 0) {
            this->nextUpdateNs = CurrentSimNanos + this->holdPeriodNs;
        }
    }

    // --- 2. Apply Dithering & Write Output ---
    // Add noise to the held base torque at every time step (e.g. 0.1s)
    double finalTorque[3];
    for(int i=0; i<3; i++) {
        double noise = 0.0;
        if (this->ditherStd > 0.0) {
            noise = this->ditherDist(this->rng) * this->ditherStd;
        }
        finalTorque[i] = this->currentBaseTorque[i] + noise;
    }

    /*! - store the output message */
    v3Copy(finalTorque, outMsgBuffer.torqueRequestBody);

    /*! - write the module output message */
    this->cmdTorqueOutMsg.write(&outMsgBuffer, this->moduleID, CurrentSimNanos);

}

void RandomTorque::setTorqueMagnitude(double value)
{
    // check that value is in acceptable range
    if (value >= 0.0) {
        this->torqueMagnitude = value;
        // Update distribution with new magnitude
        this->dist = std::uniform_real_distribution<double>(-this->torqueMagnitude, this->torqueMagnitude);
    }
}

void RandomTorque::setSeed(unsigned int value)
{
    this->seed = value;
    // Reinitialize random number generator with new seed
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