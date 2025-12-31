#ifndef RANDOM_TORQUE_H
#define RANDOM_TORQUE_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/msgPayloadDefC/CmdTorqueBodyMsgPayload.h"
#include "architecture/msgPayloadDefC/RWArrayConfigMsgPayload.h"
#include "architecture/msgPayloadDefC/RWSpeedMsgPayload.h"
#include "architecture/msgPayloadDefC/AttGuidMsgPayload.h"
#include "architecture/msgPayloadDefC/VehicleConfigMsgPayload.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/messaging/messaging.h"
#include "architecture/utilities/linearAlgebra.h"

#include <random>

// Control Mode Definitions
enum TorqueMode {
    MODE_UNIFORM = 0,   // Random values in [-Mag, +Mag]
    MODE_BANGBANG = 1,  // Either -Mag or +Mag
    MODE_LOW_MAG = 2    // Random values in [-0.1*Mag, +0.1*Mag]
};

/*! @brief Random torque generator module */
class RandomTorque: public SysModel {
public:
    RandomTorque();
    ~RandomTorque();

    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);

    Message<CmdTorqueBodyMsgPayload> cmdTorqueOutMsg;  //!< commanded torque output message
    ReadFunctor<AttGuidMsgPayload> guidInMsg;          //!< attitude guidance input message
    ReadFunctor<RWArrayConfigMsgPayload> rwParamsInMsg; //!< RW parameter input message
    ReadFunctor<RWSpeedMsgPayload> rwSpeedsInMsg;       //!< RW speed input message
    ReadFunctor<VehicleConfigMsgPayload> vehConfigInMsg; //!< vehicle configuration input message

    BSKLogger bskLogger;                                //!< BSK Logging

    /** setter for `torqueMagnitude` property */
    void setTorqueMagnitude(double value);
    /** getter for `torqueMagnitude` property */
    double getTorqueMagnitude() const {return this->torqueMagnitude;}

    /** setter for `seed` property */
    void setSeed(unsigned int value);
    /** getter for `seed` property */
    unsigned int getSeed() const {return this->seed;}

    /** New Configuration Methods */
    void setHoldPeriod(double seconds);       // Input hold period in seconds
    void setDitherStd(double value);          // Standard deviation for dithering noise
    void setControlMode(int mode);            // 0: Uniform, 1: Bang-Bang, 2: Low-Mag

private:
    double torqueMagnitude = 0.01;                     //!< [Nm] Maximum magnitude of random torque
    unsigned int seed = 0;                              //!< Random number generator seed (0 = use time-based seed)
    std::mt19937 rng;                                  //!< Random number generator
    std::uniform_real_distribution<double> dist;       //!< Uniform distribution for random torque
    double ISCPntB_B[9];                               //!< [kg m^2] Spacecraft Inertia (from vehConfigInMsg)

    // New members for Hold & Dithering logic
    uint64_t holdPeriodNs;                    //!< [ns] Hold period
    uint64_t nextUpdateNs;                    //!< [ns] Next simulation time to update base torque
    double currentBaseTorque[3];              //!< [Nm] Currently held base torque vector
    
    double ditherStd;                         //!< [Nm] Standard deviation for dithering
    int controlMode;                          //!< Control mode (TorqueMode enum)
    std::normal_distribution<double> ditherDist; //!< Normal distribution for dithering
};

#endif