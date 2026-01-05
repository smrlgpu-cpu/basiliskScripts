import os
import numpy as np
import h5py
import random
import shutil
import argparse
from datetime import datetime

# Basilisk imports
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.architecture import messaging
from Basilisk.architecture import bskLogging

# Simulation modules
from Basilisk.simulation import spacecraft
from Basilisk.simulation import exponentialAtmosphere
from Basilisk.simulation import facetDragDynamicEffector
from Basilisk.simulation import facetSRPDynamicEffector
from Basilisk.simulation import GravityGradientEffector
from Basilisk.simulation import simpleNav
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import fuelTank
from Basilisk.simulation import linearSpringMassDamper

# FSW modules
from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import inertial3D

# External modules
from Basilisk.ExternalModules import randomTorque
from Basilisk.ExternalModules import movingPulsatingBall

from Basilisk import __path__

# Monte Carlo imports
from Basilisk.utilities.MonteCarlo.Controller import Controller, RetentionPolicy
from Basilisk.utilities.MonteCarlo.Dispersions import (UniformEulerAngleMRPDispersion, NormalVectorCartDispersion)

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

# Module-level global variables for Monte Carlo (to avoid functools.partial pickling issues)
MC_CTRL_DT = 1.0
MC_LOG_DT = 0.1
MC_SIM_DT = 0.01
MC_SIM_TIME = 1000.0
MC_SLOSHING_MODEL = "none"
MC_VALIDATION = False

# Helper function to compute dcm_F0B from normal vector (must be at module level for pickling)
def normalToDcmF0B(nHat_B):
    """B frame 법선 벡터로부터 dcm_F0B 계산 (F frame +Y = nHat_B)"""
    nHat_B = np.array(nHat_B) / np.linalg.norm(nHat_B)
    if np.allclose(nHat_B, [0, 1, 0]):
        return np.eye(3)
    elif np.allclose(nHat_B, [0, -1, 0]):
        return rbk.PRV2C(np.pi * np.array([1.0, 0.0, 0.0]))
    else:
        # Gram-Schmidt로 직교 기저 구성
        y_axis = nHat_B
        temp = np.array([1.0, 0.0, 0.0]) if abs(y_axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x_axis = temp - np.dot(temp, y_axis) * y_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis) / np.linalg.norm(np.cross(x_axis, y_axis))
        return np.column_stack([x_axis, y_axis, z_axis]).T

def createScenario():
    """
    Creates the simulation scenario for Monte Carlo runs.
    Uses global variables MC_CONTROLLER_TIMESTEP and MC_SIMULATION_TIME.
    """
    global MC_CTRL_DT, MC_LOG_DT, MC_SIM_DT, MC_SIM_TIME, MC_SLOSHING_MODEL
    ctrlDtNano = macros.sec2nano(MC_CTRL_DT)
    logDtNano = macros.sec2nano(MC_LOG_DT)
    simDtNano = macros.sec2nano(MC_SIM_DT)
    simTimeNano = macros.sec2nano(MC_SIM_TIME)
    # Create simulation container
    scSim = SimulationBaseClass.SimBaseClass()
    
    # Process and Task names
    navTaskName = "navTask"
    dynTaskName = "dynTask"
    ctrlTaskName = "ctrlTask"
    logTaskName = "logTask"

    simProcessName = "simProcess"
    
    # Create process - MUST attach to scSim to prevent GC
    scSim.dynProcess = scSim.CreateNewProcess(simProcessName)
    
    # Set Simulation Time Step (Dynamics) - Faster than controller
    scSim.dynProcess.addTask(scSim.CreateNewTask(navTaskName, logDtNano))
    scSim.dynProcess.addTask(scSim.CreateNewTask(ctrlTaskName, logDtNano))
    scSim.dynProcess.addTask(scSim.CreateNewTask(logTaskName, logDtNano))
    scSim.dynProcess.addTask(scSim.CreateNewTask(dynTaskName, simDtNano))

    # --- 1. Spacecraft Setup ---
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "SMRL-Sat"
    
    # Parameters
    I = [101.67, 0., 0., 0., 135.42, 0., 0., 0., 153.75]
    scObject.hub.mHub = 500.0
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    
    # Initial Orbit
    oe = orbitalMotion.ClassicElements()
    oe.a = (6378.1366 + 402.72) * 1000
    oe.e = 0.00130547
    oe.i = 51.60 * macros.D2R
    oe.Omega = 198.38 * macros.D2R
    oe.omega = 39.26 * macros.D2R
    oe.f = 117.71 * macros.D2R
    
    # Initial Attitude
    # Generate Uniform Random Rotation using Shoemake's Method
    # u1, u2, u3 are uniform random variables in [0, 1]
    u = np.random.rand(3)
    
    # Shoemake's formula for uniform quaternion on S3
    q_rand = np.array([
        np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
        np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
        np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
        np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
    ])
    
    # Basilisk uses [q0, q1, q2, q3] where q0 is scalar
    # Shoemake's result is a valid unit quaternion.
    if q_rand[0] < 0:
        q_rand = -q_rand
    
    # Convert to MRP for initialization
    scObject.hub.sigma_BNInit = rbk.EP2MRP(q_rand)
    scObject.hub.omega_BN_BInit = np.random.uniform(-0.2, 0.2, size=(3, 1)).tolist()

    scSim.AddModelToTask(dynTaskName, scObject, 1)
    
    # Save scObject to scSim for MC dispersion
    scSim.scObject = scObject

    # --- 2. Environment Setup ---
    # Important: Attach gravFactory to scSim to prevent garbage collection
    scSim.gravFactory = simIncludeGravBody.gravBodyFactory()
    scSim.gravBodies = scSim.gravFactory.createBodies('earth', 'sun')
    scSim.earth = scSim.gravBodies['earth']
    scSim.earth.useSphericalHarmonicsGravityModel(bskPath + '/supportData/LocalGravData/GGM03S.txt', 4)
    
    mu = scSim.earth.mu
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN
    
    scSim.gravFactory.addBodiesTo(scObject)
    
    timeInitString = '2025 DECEMBER 09 00:00:00.0'
    scSim.spiceObject = scSim.gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
    scSim.spiceObject.zeroBase = 'Earth'
    scSim.AddModelToTask(dynTaskName, scSim.gravFactory.spiceObject, 2)
    
    scSim.atmo = exponentialAtmosphere.ExponentialAtmosphere()
    scSim.atmo.ModelTag = "exponentialAtmosphere"
    scSim.atmo.planetRadius = 6378136.6
    scSim.atmo.scaleHeight = 7200.0
    scSim.atmo.baseDensity = 1.217
    scSim.atmo.addSpacecraftToModel(scObject.scStateOutMsg)
    scSim.AddModelToTask(dynTaskName, scSim.atmo)

    # --- 3. Disturbances ---
    # Drag
    scSim.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
    hubDragCoeff = 2.2
    hubSize = 1.0
    hubArea = hubSize ** 2.0
    hubOffSet = hubSize / 2.0
    hubNormals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    for normal in hubNormals:
        location = normal * hubOffSet
        scSim.dragEffector.addFacet(hubArea, hubDragCoeff, normal, location)
    
    panelArea, panelCoeff = 2.0, 2.2
    panelDist = hubOffSet + 1.0
    panel_data = [
        (np.array([0, 0,  1]), np.array([0,  panelDist, 0])),
        (np.array([0, 0, -1]), np.array([0,  panelDist, 0])),
        (np.array([0, 0,  1]), np.array([0, -panelDist, 0])),
        (np.array([0, 0, -1]), np.array([0, -panelDist, 0]))
    ]
    for normal, loc in panel_data:
        scSim.dragEffector.addFacet(panelArea, panelCoeff, normal, loc)
    
    scSim.dragEffector.atmoDensInMsg.subscribeTo(scSim.atmo.envOutMsgs[0])
    scObject.addDynamicEffector(scSim.dragEffector)
    scSim.AddModelToTask(dynTaskName, scSim.dragEffector)

    # SRP
    scSim.srpEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
    scSim.srpEffector.ModelTag = "FacetSRP"
    scSim.srpEffector.setNumFacets(10)
    scSim.srpEffector.sunInMsg.subscribeTo(scSim.gravFactory.spiceObject.planetStateOutMsgs[1])
    
    # Hub SRP facets
    hubDiffuseCoeff = 0.1
    hubSpecularCoeff = 0.9
    for normal in hubNormals:
        location = normal * hubOffSet
        dcm_F0B = normalToDcmF0B(normal)
        nHat_F = np.array([0.0, 1.0, 0.0])  # F frame +Y = 법선
        rotHat_F = np.array([0.0, 0.0, 0.0])  # 회전 없음
        scSim.srpEffector.addFacet(hubArea, dcm_F0B, nHat_F, rotHat_F, location, 
                                    hubDiffuseCoeff, hubSpecularCoeff)
                            
    # 3.2.2. 태양광 패널 4면
    panelDiffuseCoeff = 0.16
    panelSpecularCoeff = 0.16
    for normal, loc in panel_data:
        dcm_F0B = normalToDcmF0B(normal)
        nHat_F = np.array([0.0, 1.0, 0.0])
        rotHat_F = np.array([0.0, 0.0, 0.0])
        scSim.srpEffector.addFacet(panelArea, dcm_F0B, nHat_F, rotHat_F, loc,
                                    panelDiffuseCoeff, panelSpecularCoeff)
        
    scObject.addDynamicEffector(scSim.srpEffector)
    scSim.AddModelToTask(dynTaskName, scSim.srpEffector)

    # Gravity Gradient
    scSim.ggEffector = GravityGradientEffector.GravityGradientEffector()
    scSim.ggEffector.ModelTag = "GravityGradient"
    scSim.ggEffector.addPlanetName(scSim.earth.planetName)
    scObject.addDynamicEffector(scSim.ggEffector)
    scSim.AddModelToTask(dynTaskName, scSim.ggEffector)

    sloshingModel = MC_SLOSHING_MODEL

    # --- Sloshing Model Selection ---
    if sloshingModel == "mpbm":
        scSim.mpbm = movingPulsatingBall.MovingPulsatingBall()
        scSim.mpbm.ModelTag = "mpbm"
        scSim.mpbm.massInit = 100.0
        scSim.mpbm.radiusTank = 0.5
        scSim.mpbm.r_TB_B = [[0.0], [0.0], [0.1]]
        
        scObject.addStateEffector(scSim.mpbm)
        scSim.AddModelToTask(dynTaskName, scSim.mpbm)

    elif sloshingModel == "spring":
        scSim.tank = fuelTank.FuelTank()
        scSim.tankModel = fuelTank.FuelTankModelConstantVolume()
        scSim.tankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        scSim.tankModel.radiusTankInit = 0.5
        scSim.particles = []

        directions = [[1,0,0], [0,1,0], [0,0,1]]
        positions = [[0.1,0,-0.1], [0,0,0.1], [-0.1,0,0.1]]
        for i, (direction, position) in enumerate(zip(directions, positions)):
            particle = linearSpringMassDamper.LinearSpringMassDamper()
            particle.k = 0.2
            particle.c = 0.05
            particle.r_PB_B = [[position[0]], [position[1]], [position[2]]]
            particle.pHat_B = [[direction[0]], [direction[1]], [direction[2]]]
            particle.rhoInit = 0.05 if i == 0 else -0.025
            particle.rhoDotInit = 0.0
            particle.massInit = 10.0
            scSim.particles.append(particle)
        
        scSim.tankModel.propMassInit = 70.0
        scSim.tank.setTankModel(scSim.tankModel)
        scSim.tank.r_TB_B = [[0], [0], [0.1]]
        scSim.tank.nameOfMassState = "fuelTankMass"
        scSim.tank.updateOnly = True
        
        for particle in scSim.particles:
            scSim.tank.pushFuelSloshParticle(particle)
            scObject.addStateEffector(particle)
            
        scObject.addStateEffector(scSim.tank)

    else:
        scSim.tank = fuelTank.FuelTank()
        scSim.tankModel = fuelTank.FuelTankModelConstantVolume()
        scSim.tankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        scSim.tankModel.radiusTankInit = 0.5
        scSim.tankModel.propMassInit = 100.0
        
        scSim.tank.setTankModel(scSim.tankModel)
        scSim.tank.r_TB_B = [[0], [0], [0.1]]
        scSim.tank.nameOfMassState = "fuelTankMass"
        scSim.tank.updateOnly = True
        
        scObject.addStateEffector(scSim.tank)

    # --- 4. Navigation & Control ---
    scSim.sNavObject = simpleNav.SimpleNav()
    scSim.sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(navTaskName, scSim.sNavObject)
    scSim.sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    # Target Attitude
    scSim.inertial3DObj = inertial3D.inertial3D()
    scSim.inertial3DObj.ModelTag = "inertial3D"
    scSim.AddModelToTask(navTaskName, scSim.inertial3DObj)
    scSim.inertial3DObj.sigma_R0N = [0., 0., 0.]

    # Attitude Error
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(navTaskName, attError)
    attError.attNavInMsg.subscribeTo(scSim.sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(scSim.inertial3DObj.attRefOutMsg)
    
    # Save for Retention
    scSim.attError = attError

    # Random Torque Control
    scSim.rngControl = randomTorque.RandomTorque()
    scSim.rngControl.ModelTag = "randomTorque"
    scSim.rngControl.setTorqueMagnitude(2)

    scSim.rngControl.setHoldPeriod(MC_CTRL_DT) 
    
    # 2. Dithering 추가 (Vanishing B 방지)
    scSim.rngControl.setDitherStd(0.01) 
    
    # 3. 모드 섞기 (데이터 다양성 확보)
    mode_distribution = [0]*4 + [1]*1 + [2]*1 + [3]*1 + [4]*1
    selected_mode = random.choice(mode_distribution)
    
    scSim.rngControl.setControlMode(int(selected_mode))
    
    scSim.AddModelToTask(ctrlTaskName, scSim.rngControl)
    
    # Connections
    scSim.vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=I)
    scSim.configDataMsg = messaging.VehicleConfigMsg().write(scSim.vehicleConfigOut)
    
    scSim.rngControl.vehConfigInMsg.subscribeTo(scSim.configDataMsg)
    scSim.rngControl.guidInMsg.subscribeTo(scSim.attError.attGuidOutMsg)
    
    # Apply Torque
    scSim.ctrlFTObject = extForceTorque.ExtForceTorque()
    scSim.ctrlFTObject.ModelTag = "controlForceTorque"
    scObject.addDynamicEffector(scSim.ctrlFTObject)
    scSim.AddModelToTask(dynTaskName, scSim.ctrlFTObject)
      
    scSim.ctrlFTObject.cmdTorqueInMsg.subscribeTo(scSim.rngControl.cmdTorqueOutMsg)
    
    # Save for Retention
    # scSim.rngControl is already set above
    
    # --- Logging Setup ---
    # Monte Carlo RetentionPolicy expects recorders to be in scSim.msgRecList
    scSim.msgRecList = {}
    
    # 1. Attitude Error Recorder
    # Key name must match the name used in RetentionPolicy.addMessageLog
    scSim.msgRecList["attError.attGuidOutMsg"] = scSim.attError.attGuidOutMsg.recorder(logDtNano)
    scSim.AddModelToTask(logTaskName, scSim.msgRecList["attError.attGuidOutMsg"])
    
    # 2. Control Torque Recorder
    scSim.msgRecList["rngControl.cmdTorqueOutMsg"] = scSim.rngControl.cmdTorqueOutMsg.recorder(logDtNano)
    scSim.AddModelToTask(logTaskName, scSim.msgRecList["rngControl.cmdTorqueOutMsg"])

    # [Validation] Recorder for MPBM Internal States
    if MC_VALIDATION and sloshingModel == "mpbm":
        scSim.msgRecList["mpbm.mpbmOutMsg"] = scSim.mpbm.mpbmOutMsg.recorder(logDtNano)
        scSim.AddModelToTask(logTaskName, scSim.msgRecList["mpbm.mpbmOutMsg"])
    
    # Store simulation time and sampling time for executeScenario
    scSim.simulationTime = simTimeNano
    scSim.samplingTime = logDtNano
    
    return scSim

def executeScenario(sim):
    # print(f"DEBUG: Initializing simulation...") 
    sim.InitializeSimulation()
    sim.ConfigureStopTime(sim.simulationTime)
    sim.ExecuteSimulation()

def check_for_nans(data):
    """
    Checks if the Monte Carlo data dictionary contains any NaN values.
    data format: {"messages": {"varName": np.array([[time, val1...], ...])}}
    """
    if "messages" not in data:
        return False
    
    for key, val in data["messages"].items():
        if np.isnan(val).any():
            return True
    return False

def run_single_retry():
    """
    Manually runs a single simulation (outside of MC controller) to replace a failed/NaN run.
    Because createScenario() uses random initialization, this provides new initial values.
    Returns: A dictionary formatted exactly like monteCarlo.getRetainedData(i).
    """
    # 1. Create a fresh scenario (Random initialization happens here)
    sim = createScenario()
    
    # 2. Execute
    executeScenario(sim)
    
    # 3. Manually extract data to match MC RetentionPolicy format
    # MC format: data["messages"]["key"] = numpy array with [time, data...]
    messages = {}
    
    # Extract attError.attGuidOutMsg (sigma_BR and omega_BR_B)
    # Note: We need to pull the specific variables recorded
    rec_att = sim.msgRecList["attError.attGuidOutMsg"]
    times = rec_att.times() # nanoseconds
    
    # Basilisk recorders return data. We need to prepend time.
    sigma_BR = unitTestSupport.addTimeColumn(times, rec_att.sigma_BR)
    omega_BR_B = unitTestSupport.addTimeColumn(times, rec_att.omega_BR_B)
    
    messages["attError.attGuidOutMsg.sigma_BR"] = sigma_BR
    messages["attError.attGuidOutMsg.omega_BR_B"] = omega_BR_B
    
    # Extract rngControl.cmdTorqueOutMsg (torqueRequestBody)
    rec_trq = sim.msgRecList["rngControl.cmdTorqueOutMsg"]
    torque = unitTestSupport.addTimeColumn(rec_trq.times(), rec_trq.torqueRequestBody)
    
    messages["rngControl.cmdTorqueOutMsg.torqueRequestBody"] = torque
    
    # [Validation] Extract MPBM Internal States
    if MC_VALIDATION and MC_SLOSHING_MODEL == "mpbm":
        rec_mpbm = sim.msgRecList["mpbm.mpbmOutMsg"]
        
        # Extract vectors and add time column
        r_slug = unitTestSupport.addTimeColumn(rec_mpbm.times(), rec_mpbm.r_BN_N)
        v_slug = unitTestSupport.addTimeColumn(rec_mpbm.times(), rec_mpbm.v_BN_N)
        torque_int = unitTestSupport.addTimeColumn(rec_mpbm.times(), rec_mpbm.omega_BN_B)
        
        messages["mpbm.mpbmOutMsg.r_BN_N"] = r_slug
        messages["mpbm.mpbmOutMsg.v_BN_N"] = v_slug
        messages["mpbm.mpbmOutMsg.omega_BN_B"] = torque_int

    return {"messages": messages}

def run_mc_generation(*, numRuns: int, ctrlDt: float, logDt: float, simDt: float, simTime: float, numThreads: int = 4, sloshingModel: str = "none", validation: bool = False):
    # Suppress Basilisk INFO messages (only show WARNING and above)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)
    
    # Set global variables for createScenario (avoid functools.partial pickling issues)
    global MC_CTRL_DT, MC_LOG_DT, MC_SIM_DT, MC_SIM_TIME, MC_SLOSHING_MODEL, MC_VALIDATION
    MC_CTRL_DT = ctrlDt
    MC_LOG_DT = logDt
    MC_SIM_DT = simDt
    MC_SIM_TIME = simTime
    MC_SLOSHING_MODEL = sloshingModel
    MC_VALIDATION = validation
    
    # Setup directories
    experimentBaseDir = os.path.join("data", "experiments", fileName)
    os.makedirs(experimentBaseDir, exist_ok=True)
    
    rawBaseDir = os.path.join("data", "raw")
    os.makedirs(rawBaseDir, exist_ok=True)
    
    datetimeStr = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Monte Carlo Controller
    monteCarlo = Controller()
    monteCarlo.setSimulationFunction(createScenario)
    monteCarlo.setExecutionFunction(executeScenario)
    monteCarlo.setExecutionCount(numRuns)
    monteCarlo.setShouldDisperseSeeds(True)
    monteCarlo.setThreadCount(numThreads)
    monteCarlo.setVerbose(False)
    monteCarlo.setShowProgressBar(True)
    monteCarlo.setArchiveDir(os.path.join(experimentBaseDir, datetimeStr))
    
    # Retention Policy
    retentionPolicy = RetentionPolicy()
    logDtNano = macros.sec2nano(logDt)
    retentionPolicy.logRate = logDtNano
    
    retentionPolicy.addMessageLog("attError.attGuidOutMsg", ["sigma_BR", "omega_BR_B"])
    retentionPolicy.addMessageLog("rngControl.cmdTorqueOutMsg", ["torqueRequestBody"])

    if validation and sloshingModel == "mpbm":
        retentionPolicy.addMessageLog("mpbm.mpbmOutMsg", ["r_BN_N", "v_BN_N", "omega_BN_B"])
    
    monteCarlo.addRetentionPolicy(retentionPolicy)
    
    # Execute
    print(f"Starting Monte Carlo simulation with {numRuns} runs...")
    failures = monteCarlo.executeSimulations()
    if failures:
        print(f"Failed runs reported by MC controller: {failures}")
        
    # Data Collection & HDF5 Saving
    ctrlSeqLen = int(round(simTime / ctrlDt))
    
    # Filename format per request
    if sloshingModel == "none":
        h5_filename = f"attitude_{numRuns}_{ctrlSeqLen}_{logDt}.h5"
    elif sloshingModel == "mpbm":
        h5_filename = f"mpbm_{numRuns}_{ctrlSeqLen}_{logDt}.h5"
    elif sloshingModel == "spring":
        h5_filename = f"spring_{numRuns}_{ctrlSeqLen}_{logDt}.h5"
    else:
        h5_filename = f"{sloshingModel}_{numRuns}_{ctrlSeqLen}_{logDt}.h5"

    h5_path = os.path.join(rawBaseDir, h5_filename)
    
    print(f"Saving data to {h5_path}...")
    
    with h5py.File(h5_path, 'w') as f:
        grp_ts = f.create_group("timeseries")
        
        for i in range(numRuns):
            # Retrieve data from MC archive
            data = monteCarlo.getRetainedData(i)
            
            # --- NaN Check & Retry Logic ---
            retry_count = 0
            max_retries = 10  # Prevent infinite loops
            
            while check_for_nans(data) and retry_count < max_retries:
                print(f"WARNING: NaN detected in Run {i}. Retrying with new initialization (Attempt {retry_count + 1})...")
                
                # Run a single clean simulation manually
                data = run_single_retry()
                
                retry_count += 1
                
            if check_for_nans(data):
                print(f"ERROR: Run {i} failed to produce clean data after {max_retries} retries. Saving dirty data.")
            # -------------------------------
            
            grp_seq = grp_ts.create_group(f"sequence_{i}")
            
            # Extract data
            sigma_mrp_data = data["messages"]["attError.attGuidOutMsg.sigma_BR"][:, 1:]
            omega = data["messages"]["attError.attGuidOutMsg.omega_BR_B"][:, 1:]
            torque = data["messages"]["rngControl.cmdTorqueOutMsg.torqueRequestBody"][:, 1:]
            
            # Convert MRP to Quaternion (EP)
            num_steps = sigma_mrp_data.shape[0]
            quaternion_data = np.zeros((num_steps, 4))
            
            for k in range(num_steps):
                quaternion_data[k] = rbk.MRP2EP(sigma_mrp_data[k])
            
            # Combine state: [quaternion (4), omega (3)]
            state = np.hstack((quaternion_data, omega))
            
            # Handle torque length mismatch
            if torque.shape[0] < state.shape[0]:
                diff = state.shape[0] - torque.shape[0]
                padding = np.zeros((diff, 3))
                torque = np.vstack((torque, padding))
            elif torque.shape[0] > state.shape[0]:
                torque = torque[:state.shape[0]]
            
            # Save datasets
            grp_seq.create_dataset("state", data=state, dtype=np.float32)
            grp_seq.create_dataset("control_torque", data=torque, dtype=np.float32)

            if validation and sloshingModel == "mpbm":
                r_slug = data["messages"]["mpbm.mpbmOutMsg.r_BN_N"][:, 1:]
                v_slug = data["messages"]["mpbm.mpbmOutMsg.v_BN_N"][:, 1:]
                torque_int = data["messages"]["mpbm.mpbmOutMsg.omega_BN_B"][:, 1:]
                
                grp_seq.create_dataset("slug_position", data=r_slug, dtype=np.float32)
                grp_seq.create_dataset("slug_velocity", data=v_slug, dtype=np.float32)
                grp_seq.create_dataset("interaction_torque", data=torque_int, dtype=np.float32)
            
    print("Data generation complete.")
    
    # Clean up MC temp data
    if os.path.exists(monteCarlo.archiveDir):
        shutil.rmtree(monteCarlo.archiveDir)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate attitude control data using Monte Carlo simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulation runs (sequences)'
    )
    parser.add_argument(
        '--ctrl-dt',
        type=float,
        default=1.0,
        help='Controller timestep in seconds (default: 1.0s)'
    )
    parser.add_argument(
        '--log-dt',
        type=float,
        default=0.1,
        help='Logging timestep in seconds (default: 0.1s)'
    )
    parser.add_argument(
        '--sim-dt',
        type=float,
        default=0.01,
        help='Simulation timestep in seconds (default: 0.01s)'
    )
    parser.add_argument(
        '--sim-time',
        type=float,
        default=1000.0,
        help='Total simulation time for each sequence in seconds (default: 1000.0s)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=16,
        help='Number of parallel threads for Monte Carlo execution'
    )
    parser.add_argument(
        '--sloshing',
        type=str,
        default='none',
        choices=['none', 'spring', 'mpbm'],
        help='Sloshing model to use: none, spring, or mpbm (default: none)'
    )
    parser.add_argument(
        '--validation',
        action='store_true',
        help='Log internal states for validation (only for mpbm model)'
    )
    args = parser.parse_args()
    
    # Run data generation
    print(f"Starting data generation with:")
    print(f"  - Number of runs: {args.num_runs}")
    print(f"  - Controller timestep: {args.ctrl_dt}s ")
    print(f"  - Logging timestep: {args.log_dt}s ")
    print(f"  - Simulation timestep: {args.sim_dt}s ")
    print(f"  - Simulation time: {args.sim_time}s")
    print(f"  - Parallel threads: {args.threads}")
    print(f"  - Sloshing model: {args.sloshing}")
    print()
    
    run_mc_generation(
        numRuns=args.num_runs, 
        ctrlDt=args.ctrl_dt, 
        logDt=args.log_dt, 
        simDt=args.sim_dt, 
        simTime=args.sim_time, 
        numThreads=args.threads,
        sloshingModel=args.sloshing,
        validation=args.validation
    )
