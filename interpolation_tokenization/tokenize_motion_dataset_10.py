import os, gc, math, json, argparse, pickle, random, shutil
from pathlib import Path
from collections import Counter
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm

# ============================================================
# SCRIPT 2: Tokenizer
# This script reads the INTERPOLATED dataset (with motion vectors)
# and converts the (T, K, 2) motion vectors into (T, K) token tensors.
#
# This version is modified to accept motion vectors with
# values like {-10, 0, 10} and convert them for tokenization.
# ============================================================

def _safe_write_pickle(obj, dest_path: str):
    # We still use /tmp for the temporary file to avoid disk quota errors
    tmp_dir = Path(os.getenv("TEMP", "/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = str(tmp_dir / f"_token_tmp_{os.getpid()}_{random.randint(0,1_000_000)}.pkl")
    
    try:
        print(f"Writing temporary file to: {tmp_file}")
        with open(tmp_file, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Moving temporary file to final destination: {dest_path}")
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
        
        shutil.move(tmp_file, dest_path)
        print("Move complete.")

    except Exception as e:
        print(f"Error during safe write: {e}")
        if os.path.exists(tmp_file):
            print(f"Cleaning up temporary file: {tmp_file}")
            os.remove(tmp_file)
        raise

# -------- Runner (in-place dict, batch loop) --------
def run(input_pkl: str, output_pkl: str, kpoints: int = 133, axis_step: float = 1.0):
    
    print(f"Loading motion vector dataset: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data: Dict[str, dict] = pickle.load(f)
    print(f"Loaded. Entries: {len(data):,}")

    keys = list(data.keys())
    N = len(keys)
    
    # This is the step size 's' used in the interpolation script
    s = axis_step
    if s == 1.0:
        print("Warning: axis_step=1.0. Running in original {-1,0,1} mode.")
    else:
        print(f"Using axis_step={s}. Will convert {-s, 0, s} -> {-1, 0, 1}.")
    
    # --- This is our GPU-based setup ---
    
    # 0. Check for GPU and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: No GPU found. This will be very slow.")

    # 1. Create the 9-token lookup table ONCE, on the GPU
    # This maps our (dx, dy) pairs to your token IDs (1-9)
    #
    # Our math will create indices 0-8 like this:
    # idx 0: (-1, -1) -> token 9
    # idx 1: (-1, 0)  -> token 7  
    # idx 2: (-1, 1)  -> token 8
    # idx 3: (0, -1)  -> token 3
    # idx 4: (0, 0)   -> token 1
    # idx 5: (0, 1)   -> token 2
    # idx 6: (1, -1)  -> token 6
    # idx 7: (1, 0)   -> token 4
    # idx 8: (1, 1)   -> token 5
    lookup_table = torch.tensor(
        [9, 7, 8, 3, 1, 2, 6, 4, 5], 
        dtype=torch.uint8, 
        device=device
    )
    # --- End new setup ---

    print("Starting fast GPU tokenization loop...")
    for i in tqdm(range(N)):
        k = keys[i]
        entry = data.get(k, None)
        if not (isinstance(entry, dict) and "motion_vector" in entry):
            continue

        # Load the (T, K, 2) motion vector
        motion_vector_3d = entry["motion_vector"]

        # 2. Ensure it's a PyTorch tensor and move to GPU
        if isinstance(motion_vector_3d, np.ndarray):
            mv = torch.from_numpy(motion_vector_3d).to(device)
        else:
            # If it's already a tensor, just move it
            mv = motion_vector_3d.to(device)
        
        T, K, D = mv.shape
        if K != kpoints or D != 2:
            print(f"Skipping {k}: expected shape (T, {kpoints}, 2), got ({T}, {K}, {D})")
            continue

        # --- THIS IS THE MODIFIED LOGIC ---
        
        # 3. Separate dx, dy and cast to long for math
        # Shape of dx and dy will be (T, K)
        # We use .long() because tensor indexing requires long integers
        dx = mv[..., 0].long()
        dy = mv[..., 1].long()
        
        # 4. Map values from {-s, 0, s} to indices {0, 1, 2}
        #    (dx / s) converts {-10, 0, 10} -> {-1, 0, 1}
        #    + 1 converts {-1, 0, 1} -> {0, 1, 2}
        dx_mapped = (dx / s).long() + 1
        dy_mapped = (dy / s).long() + 1
        
        # --- END MODIFIED LOGIC ---

        # 5. Create the flat index (0-8) using "base 3" math
        # This (T,K) tensor now has values from 0 to 8
        flat_indices = (dx_mapped * 3) + dy_mapped
        
        # 6. Use the indices to "gather" tokens from the lookup table
        # This is one, single, ultra-fast operation.
        # tokenized_tensor will be on the GPU.
        tokenized_tensor = lookup_table[flat_indices]
        
        # --- Update the dictionary ---
        # Replace the (T,K,2) motion vector with the (T,K) token tensor
        del entry["motion_vector"] # Free memory
        
        # Move back to CPU and convert to NumPy for pickling
        # We use uint8 to save 4x memory vs int32 for tokens 1-9
        entry["motion_tokens"] = tokenized_tensor.cpu().numpy()
        # num_frames, name, and gloss are already correct
    
    gc.collect()
    print(f"Tokenization complete for {N} entries.")

    # Robust write to new file
    print(f"Writing final TOKENIZED dataset → {output_pkl}")
    _safe_write_pickle(data, output_pkl)
    
    print("\n=== DONE ===")
    print(f"Output tokenized dataset : {output_pkl}")
    print("=====================================================")

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Phoenix motion vector to token tensor conversion.")
    ap.add_argument("input",  help="Path to *interpolated* pickle (.pkl) with motion vectors")
    ap.add_argument("output", help="Path to write *tokenized* pickle (.pkl)")
    ap.add_argument("--kpoints", type=int, default=133, help="Keypoints per frame (default 133 for MSKA)")
    # --- NEW REQUIRED ARGUMENT ---
    ap.add_argument("--axis_step", type=float, default=1.0, 
                    help="[REQUIRED] Must match the --axis_step used in the interpolation script (e.g., 10.0)")
    args = ap.parse_args()

    if args.kpoints != 133:
        print(f"Warning: You set --kpoints={args.kpoints}, but data may have 133. Mismatch may occur.")

    if args.axis_step == 1.0:
        print("Warning: Running with --axis_step=1.0. This is for the original {-1,0,1} motion vectors.")
    elif args.axis_step <= 0:
        print("Error: --axis_step must be a positive number.")
        return
    else:
        print(f"Running with --axis_step={args.axis_step}. Expecting inputs of {-args.axis_step, 0, args.axis_step}.")


    run(
        args.input,
        args.output,
        kpoints=args.kpoints,
        axis_step=args.axis_step,
    )

if __name__ == "__main__":
    main()