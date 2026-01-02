%module randomTorque

%include "architecture/utilities/bskException.swg"
%default_bsk_exception();

%{
    #include "randomTorque.h"
%}

%pythoncode %{
from Basilisk.architecture.swig_common_model import *
%}

%include "sys_model.i"
%include "swig_conly_data.i"
%include "std_string.i"

%include "randomTorque.h"

// 메시지 페이로드 정의
%include "architecture/msgPayloadDefC/CmdTorqueBodyMsgPayload.h"
struct CmdTorqueBodyMsg_C;
%include "architecture/msgPayloadDefC/AttGuidMsgPayload.h"
struct AttGuidMsg_C;
%include "architecture/msgPayloadDefC/VehicleConfigMsgPayload.h"
struct VehicleConfigMsg_C;

%pythoncode %{
import sys
protectAllClasses(sys.modules[__name__])
%}