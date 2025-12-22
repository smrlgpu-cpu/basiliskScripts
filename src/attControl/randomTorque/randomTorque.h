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

private:
    double torqueMagnitude = 0.01;                     //!< [Nm] Maximum magnitude of random torque
    unsigned int seed = 0;                              //!< Random number generator seed (0 = use time-based seed)
    std::mt19937 rng;                                  //!< Random number generator
    std::uniform_real_distribution<double> dist;       //!< Uniform distribution for random torque
    double ISCPntB_B[9];                               //!< [kg m^2] Spacecraft Inertia (from vehConfigInMsg)

};

#endif
