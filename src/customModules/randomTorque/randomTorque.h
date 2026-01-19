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
#include <vector>
#include <cmath>

/*! @brief Random torque generator module */
class RandomTorque: public SysModel {
public:
    RandomTorque();
    ~RandomTorque();

    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);

    Message<CmdTorqueBodyMsgPayload> cmdTorqueOutMsg;
    ReadFunctor<AttGuidMsgPayload> guidInMsg;
    ReadFunctor<RWArrayConfigMsgPayload> rwParamsInMsg;
    ReadFunctor<RWSpeedMsgPayload> rwSpeedsInMsg;
    ReadFunctor<VehicleConfigMsgPayload> vehConfigInMsg;

    BSKLogger bskLogger;

    void setTorqueMagnitude(double value);
    double getTorqueMagnitude() const {return this->torqueMagnitude;}

    void setSeed(unsigned int value);
    unsigned int getSeed() const {return this->seed;}

    void setHoldPeriod(double seconds); // Used as base period

private:
    double torqueMagnitude = 0.01;
    unsigned int seed = 0;
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist; // Standard [0, 1]
    double ISCPntB_B[9];

    uint64_t holdPeriodNs;
    uint64_t nextUpdateNs;
    
    double currentFinalTorque[3]; 
};

#endif