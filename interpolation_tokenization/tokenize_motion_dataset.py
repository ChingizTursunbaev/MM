#tokenize_motion_dataset.py
import os, gc, math, json, argparse, pickle, random, shutil
from pathlib import Path
from collections import Counter
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm

# ============================================================
# This script reads your INTERPOLATED dataset (with motion vectors)
# and converts the (T, K, 2) motion vectors into (T, K) token tensors.
# This is the "Step 3" pre-processing that fixes the dataloader bottleneck.
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
def run(input_pkl: str, output_pkl: str, kpoints: int = 133):
    
    print(f"Loading motion vector dataset: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data: Dict[str, dict] = pickle.load(f)
    print(f"Loaded. Entries: {len(data):,}")

    keys = list(data.keys())
    N = len(keys)
    
    # --- This is our new GPU-based setup ---
    
    # 0. Check for GPU and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

        # --- This is the fast, vectorized replacement ---
        
        # 3. Separate dx, dy and cast to long for math
        # Shape of dx and dy will be (T, K)
        # We use .long() because tensor indexing requires long integers
        dx = mv[..., 0].long()
        dy = mv[..., 1].long()
        
        # 4. Map values from {-1, 0, 1} to indices {0, 1, 2}
        dx_mapped = dx + 1
        dy_mapped = dy + 1
        
        # 5. Create the flat index (0-8) using "base 3" math
        # This (T,K) tensor now has values from 0 to 8
        flat_indices = (dx_mapped * 3) + dy_mapped
        
        # 6. Use the indices to "gather" tokens from the lookup table
        # This is one, single, ultra-fast operation.
        # tokenized_tensor will be on the GPU.
        tokenized_tensor = lookup_table[flat_indices]
        
        # --- End fast replacement ---
        
        # --- Update the dictionary ---
        # Replace the (T,K,2) motion vector with the (T,K) token tensor
        del entry["motion_vector"] # Free memory
        
        # Move back to CPU and convert to NumPy for pickling
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
    args = ap.parse_args()

    if args.kpoints != 133:
        print(f"Warning: You set --kpoints={args.kpoints}, but data may have 133. Mismatch may occur.")

    run(
        args.input,
        args.output,
        kpoints=args.kpoints,
    )

if __name__ == "__main__":
    main()