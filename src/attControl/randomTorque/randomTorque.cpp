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
    } else {
    }

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
    double randomTorque[3];                          /*!< [Nm] Random torque vector */
    CmdTorqueBodyMsgPayload outMsgBuffer;           /*!< local output message copy */
    AttGuidMsgPayload guidInMsgBuffer;               /*!< local copy of guidance input message */
    // RWArrayConfigMsgPayload rwParamsInMsgBuffer;     /*!< local copy of RW parameter input message */
    // RWSpeedMsgPayload rwSpeedsInMsgBuffer;           /*!< local copy of RW speed input message */
    // VehicleConfigMsgPayload vehConfigInMsgBuffer;     /*!< local copy of vehicle configuration input message */

    // always zero the output buffer first
    outMsgBuffer = this->cmdTorqueOutMsg.zeroMsgPayload;

    /*! - Read the optional input messages */
    if (this->guidInMsg.isLinked()) {
        guidInMsgBuffer = this->guidInMsg();
        // Guidance message is available but not used for random torque generation
        // This is just to match the interface of mrpFeedback
    }

    /*! - Generate random torque for each axis */
    randomTorque[0] = this->dist(this->rng);
    randomTorque[1] = this->dist(this->rng);
    randomTorque[2] = this->dist(this->rng);

    /*! - store the output message */
    v3Copy(randomTorque, outMsgBuffer.torqueRequestBody);

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
    } else {
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
