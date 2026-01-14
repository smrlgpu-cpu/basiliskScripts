import os
import h5py
import numpy as np
import argparse
from Basilisk.utilities import RigidBodyKinematics as rbk
import scripts.dataGeneration as dg

# Configure argument parser
parser = argparse.ArgumentParser(description="Patch HDF5 data by re-running failed simulations.")

parser.add_argument("--num-runs", type=int, default=50000, help="Number of runs (default: 50000)")
parser.add_argument("--ctrl-dt", type=float, default=1.0, help="Control time step (default: 1.0)")
parser.add_argument("--log-dt", type=float, default=0.1, help="Logging time step (default: 0.1)")
parser.add_argument("--sim-dt", type=float, default=0.0001, help="Simulation time step (default: 0.0001)")
parser.add_argument("--sim-time", type=float, default=200.0, help="Simulation duration (default: 200.0)")
parser.add_argument("--sloshing-model", type=str, default="mpbm", help="Sloshing model name (default: mpbm)")
parser.add_argument("--validation", action="store_true", help="Enable validation mode (default: False)")

args = parser.parse_args()

# Configuration from arguments
NUM_RUNS = args.num_runs
CTRL_DT = args.ctrl_dt
LOG_DT = args.log_dt
SIM_DT = args.sim_dt
SIM_TIME = args.sim_time
SLOSHING_MODEL = args.sloshing_model
VALIDATION = args.validation

# Set global variables in the dataGeneration module
dg.MC_CTRL_DT = CTRL_DT
dg.MC_LOG_DT = LOG_DT
dg.MC_SIM_DT = SIM_DT
dg.MC_SIM_TIME = SIM_TIME
dg.MC_SLOSHING_MODEL = SLOSHING_MODEL
dg.MC_VALIDATION = VALIDATION

# File path
ctrlSeqLen = int(round(SIM_TIME / CTRL_DT))
h5_filename = f"{SLOSHING_MODEL}_{NUM_RUNS}_{ctrlSeqLen}_{LOG_DT}.h5"
h5_path = os.path.join("data", "raw", h5_filename)

def patch_data():
    if not os.path.exists(h5_path):
        print(f"Error: File {h5_path} not found.")
        return

    print(f"Opening {h5_path} for patching...")
    
    with h5py.File(h5_path, 'r+') as f:
        grp_ts = f["timeseries"]
        
        # We can iterate through keys, or assume 0 to NUM_RUNS-1
        # Iterating keys is safer in case some are missing or named differently
        run_keys = list(grp_ts.keys())
        # Sort by run number to be systematic
        run_keys.sort(key=lambda x: int(x.split('_')[1]))
        
        fixed_count = 0
        
        for key in run_keys:
            # key is like "sequence_0"
            run_idx = int(key.split('_')[1])
            grp_seq = grp_ts[key]
            
            # Check for NaNs in existing data
            has_nan = False
            for dset_name in grp_seq.keys():
                data = grp_seq[dset_name][:]
                if np.isnan(data).any():
                    has_nan = True
                    break
            
            if has_nan:
                print(f"Run {run_idx} ({key}) contains NaNs. Retrying...")
                
                # Retry loop
                success = False
                for attempt in range(1, 21): # Try up to 20 times
                    print(f"  Attempt {attempt} for Run {run_idx}...")
                    try:
                        new_data = dg.run_single_retry()
                        if not dg.check_for_nans(new_data):
                            success = True
                            print(f"  Success on attempt {attempt}.")
                            
                            # Update datasets
                            # Extract data similar to run_mc_generation
                            sigma_mrp = new_data["messages"]["attError.attGuidOutMsg.sigma_BR"][:, 1:]
                            omega = new_data["messages"]["attError.attGuidOutMsg.omega_BR_B"][:, 1:]
                            torque = new_data["messages"]["rngControl.cmdTorqueOutMsg.torqueRequestBody"][:, 1:]
                            
                            num_steps = sigma_mrp.shape[0]
                            quaternion = np.zeros((num_steps, 4))
                            for k in range(num_steps):
                                quaternion[k] = rbk.MRP2EP(sigma_mrp[k])
                            
                            state = np.hstack((quaternion, omega))
                            if torque.shape[0] < state.shape[0]:
                                torque = np.vstack((torque, np.zeros((state.shape[0] - torque.shape[0], 3))))
                            elif torque.shape[0] > state.shape[0]:
                                torque = torque[:state.shape[0]]
                                
                            # Delete old datasets and create new ones (or overwrite)
                            del grp_seq["state"]
                            del grp_seq["control_torque"]
                            
                            grp_seq.create_dataset("state", data=state, dtype=np.float32)
                            grp_seq.create_dataset("control_torque", data=torque, dtype=np.float32)

                            if VALIDATION and SLOSHING_MODEL == "mpbm":
                                r_slug = new_data["messages"]["mpbm.mpbmOutMsg.r_Slug_B"][:, 1:]
                                v_slug = new_data["messages"]["mpbm.mpbmOutMsg.v_Slug_B"][:, 1:]
                                torque_int = new_data["messages"]["mpbm.mpbmOutMsg.T_Interaction"][:, 1:]
                                
                                if "slug_position" in grp_seq: del grp_seq["slug_position"]
                                if "slug_velocity" in grp_seq: del grp_seq["slug_velocity"]
                                if "interaction_torque" in grp_seq: del grp_seq["interaction_torque"]

                                grp_seq.create_dataset("slug_position", data=r_slug, dtype=np.float32)
                                grp_seq.create_dataset("slug_velocity", data=v_slug, dtype=np.float32)
                                grp_seq.create_dataset("interaction_torque", data=torque_int, dtype=np.float32)
                            
                            fixed_count += 1
                            break
                    except Exception as e:
                        print(f"  Error during retry: {e}")
                
                if not success:
                    print(f"  Failed to fix Run {run_idx} after 20 attempts.")

    print(f"Patching complete. Fixed {fixed_count} runs.")

if __name__ == "__main__":
    patch_data()

