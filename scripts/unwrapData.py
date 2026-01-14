import h5py
import numpy as np
import os
import argparse
import sys

def unwrap_quaternions(file_path):
    """
    Unwraps quaternion trajectories in the given HDF5 file to ensure continuity.
    Assumes structure: timeseries/sequence_{id}/state
    State format: [q0, q1, q2, q3, wx, wy, wz, ...]
    """
    # Construct full path assuming it's in data/raw/
    full_path = os.path.join("data", "raw", file_path)
    
    if not os.path.exists(full_path):
        print(f"Error: File not found at {full_path} or {file_path}")
        return

    print(f"Processing file: {full_path}")

    try:
        with h5py.File(full_path, 'r+') as f:
            if 'timeseries' not in f:
                print("Error: 'timeseries' group not found in the file.")
                return

            timeseries = f['timeseries']
            
            # Get list of sequences and sort them
            seq_keys = list(timeseries.keys())
            # Sort by sequence ID number (sequence_0, sequence_1, ...)
            try:
                seq_keys.sort(key=lambda x: int(x.split('_')[1]))
            except:
                print("Warning: Could not sort keys numerically. Processing in default order.")

            fixed_count = 0
            total_sequences = len(seq_keys)

            print(f"Found {total_sequences} sequences. Starting unwrap process...")

            for i, key in enumerate(seq_keys):
                group = timeseries[key]
                
                if 'state' not in group:
                    continue

                # Read state data
                # Shape is typically (Time, 7) or (Time, N)
                data = group['state'][:]
                
                # Check dimensions
                if data.shape[1] < 4:
                    print(f"Skipping {key}: State dimension too small ({data.shape})")
                    continue

                # Extract quaternions (first 4 columns)
                quats = data[:, 0:4]
                
                # Perform unwrap
                # Algorithm: If dot(q_t, q_{t-1}) < 0, then q_t = -q_t
                flips_made = 0
                for t in range(1, len(quats)):
                    # Dot product between previous (already unwrapped) and current
                    dot_prod = np.dot(quats[t-1], quats[t])
                    
                    if dot_prod < 0:
                        quats[t] *= -1.0
                        flips_made += 1
                
                # Update data only if changes were made (optimization)
                if flips_made > 0:
                    data[:, 0:4] = quats
                    
                    # Delete old dataset and write new one to ensure type/shape consistency
                    # or overwrite in place
                    group['state'][:] = data
                    fixed_count += 1
                
                # Progress indicator
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{total_sequences} sequences...")

            print(f"\nUnwrapping complete.")
            print(f"Modified {fixed_count} out of {total_sequences} sequences.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unwrap quaternion data in HDF5 files to fix sign discontinuities.")
    parser.add_argument("-f", "--file-path", type=str, help="Path to the HDF5 file (relative to data/raw/)")
    
    args = parser.parse_args()
    unwrap_quaternions(args.file_path)