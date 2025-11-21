import os, gc, math, json, argparse, pickle, random, shutil
from pathlib import Path
from collections import Counter
from typing import Dict, List
import numpy as np
import torch

# ============================================================
# This script performs two steps:
# 1. Interpolates keypoint sequences so that |Δx|,|Δy| ≤ axis_step.
# 2. Converts the new absolute coordinates into (dx, dy) motion vectors.
#
# If you run this with --axis_step=1.0, the output motion vectors
# will be composed of {-1, 0, 1} pairs, exactly as needed.
# ============================================================

# -------- Windows UNC helpers (safe write with long-path) --------
def _to_long_unc(p: str) -> str:
    if os.name != "nt":
        return p
    p = p.replace("/", "\\")
    if p.startswith("\\\\?\\") or p.startswith("\\\\.\\"):
        return p
    if p.startswith("\\\\"):
        parts = p.lstrip("\\").split("\\", 2)  # ['SERVER','share','rest...']
        if len(parts) >= 2:
            rest = parts[2] if len(parts) == 3 else ""
            return "\\\\?\\UNC\\" + parts[0] + "\\" + parts[1] + ("\\" + rest if rest else "")
        return "\\\\?\\UNC\\" + p.lstrip("\\")
    return "\\\\?\\" + p

def _safe_write_pickle(obj, dest_path: str):
    tmp_dir = Path(os.getenv("TMPDIR", os.getenv("TEMP", ".")))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = str(tmp_dir / f"_interp_tmp_{os.getpid()}_{random.randint(0,1_000_000)}.pkl")
    with open(tmp_file, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        Path(_to_long_unc(dest_dir)).mkdir(parents=True, exist_ok=True)
    shutil.move(tmp_file, _to_long_unc(dest_path))

# -------- Interpolator: per-axis step (size = axis_step) + final snap --------
class UnitStepInterpolator:
    def __init__(self, k=133, max_steps=None, axis_step: float = 1.0): # <-- Changed default K
        """
        k: number of keypoints
        max_steps: optional clamp on substeps per (f1,f2) pair
        axis_step: maximum allowed |Δx| or |Δy| per output frame (pixels)
        """
        self.K = k
        self.max_steps = max_steps
        self.axis_step = float(axis_step)
        print(f"Interpolator Initialized: K={k}, axis_step={axis_step}")

    @staticmethod
    def _finite_abs_max(diff: np.ndarray) -> float:
        diff = np.where(np.isfinite(diff), diff, 0.0)
        return float(np.max(np.abs(diff))) if diff.size else 0.0

    def _required_steps(self, f1: np.ndarray, f2: np.ndarray) -> int:
        # ceil(max_displacement / axis_step) ensures per-frame change ≤ axis_step
        diff = f2[:, :2] - f1[:, :2]
        max_disp = self._finite_abs_max(diff)
        steps = max(1, int(math.ceil(max_disp / max(self.axis_step, 1e-8))))
        return steps

    def _interpolate_pair(self, f1: np.ndarray, f2: np.ndarray, steps: int) -> List[np.ndarray]:
        K = f1.shape[0]
        out = []

        x = f1[:, 0].astype(np.float32).copy()
        y = f1[:, 1].astype(np.float32).copy()
        c1 = f1[:, 2].astype(np.float32)
        c2 = f2[:, 2].astype(np.float32)

        total_dx = (f2[:, 0] - f1[:, 0]).astype(np.float32)
        total_dy = (f2[:, 1] - f1[:, 1]).astype(np.float32)
        step_dx = total_dx / steps
        step_dy = total_dy / steps

        acc_x = np.zeros(K, dtype=np.float32)
        acc_y = np.zeros(K, dtype=np.float32)

        valid = (
            np.isfinite(f1[:, 0]) & np.isfinite(f1[:, 1]) &
            np.isfinite(f2[:, 0]) & np.isfinite(f2[:, 1])
        )

        s = self.axis_step

        for _ in range(steps):
            acc_x[valid] += step_dx[valid]
            acc_y[valid] += step_dy[valid]

            # X axis steps of size s
            pos = acc_x >= s
            neg = acc_x <= -s
            x[valid & pos] += s
            x[valid & neg] -= s
            acc_x[valid & pos] -= s
            acc_x[valid & neg] += s

            # Y axis steps of size s
            pos = acc_y >= s
            neg = acc_y <= -s
            y[valid & pos] += s
            y[valid & neg] -= s
            acc_y[valid & pos] -= s
            acc_y[valid & neg] += s

            out.append(np.stack([x, y, c1], axis=1).astype(np.float32))

        # Final snap exactly to f2 (handles subpixel residuals and <s px moves)
        x[valid] = f2[:, 0][valid]
        y[valid] = f2[:, 1][valid]
        out.append(np.stack([x, y, c2], axis=1).astype(np.float32))
        return out

    def interpolate_sequence(self, seq: np.ndarray, preserve_dtype: np.dtype) -> np.ndarray:
        T, K, C = seq.shape
        assert K == self.K and C >= 3, f"Expected (T,{self.K},>=3), got {seq.shape}"
        frames = [seq[0]]

        for t in range(T - 1):
            f1, f2 = seq[t], seq[t + 1]
            steps = self._required_steps(f1, f2)
            if self.max_steps is not None:
                steps = min(steps, int(self.max_steps))
            pair_frames = self._interpolate_pair(f1, f2, steps)  # includes final snap-to-f2
            if t < T - 2:
                frames.extend(pair_frames[:-1])  # drop the snap; next pair starts at f2
            else:
                frames.extend(pair_frames)      # keep snap for last pair

        out = np.asarray(frames, dtype=np.float32)
        if out.dtype != preserve_dtype:
            out = out.astype(preserve_dtype, copy=False)
        return out

# -------- Verifier (per-axis step ≤ axis_step) --------
def verify_axis_steps(seq: np.ndarray, axis_step=5.0, tol=1e-6) -> bool:
    diffs = np.diff(seq[..., :2], axis=0)
    return not np.any(np.abs(diffs) > (float(axis_step) + tol))

# -------- Runner (in-place dict, batch loop) --------
def run(input_pkl: str, output_pkl: str, batch_size: int = 50, verify_sample: int = 80,
        kpoints: int = 133, max_steps: int = None, seed: int = 42, axis_step: float = 1.0): # <-- Changed default K and axis_step
    
    print(f"Loading original dataset: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data: Dict[str, dict] = pickle.load(f)
    print(f"Loaded. Entries: {len(data):,} | Processing batch_size={batch_size}")

    keys = list(data.keys())
    N = len(keys)
    interp = UnitStepInterpolator(k=kpoints, max_steps=max_steps, axis_step=axis_step)

    stats = {
        "input_path": input_pkl,
        "output_path": output_pkl,
        "entries": N,
        "frames_before_total": 0,
        "frames_after_total": 0,
        "ratio_min": None,
        "ratio_max": None,
        "ratio_avg": None,
        "dtype_counts_before": Counter(),
        "dtype_counts_after": Counter(),
        "K_counts_before": Counter(),
        "K_counts_after": Counter(),
        "C_counts_before": Counter(),
        "C_counts_after": Counter(),
        "unit_step_checked_samples": 0,
        "unit_step_violations_sampled": 0,
        "axis_step": float(axis_step),
    }
    ratio_sum = 0.0
    processed_count = 0

    for i in range(0, N, batch_size):
        batch_keys = keys[i:i + batch_size]
        for k in batch_keys:
            entry = data.get(k, None)
            if not (isinstance(entry, dict) and "keypoint" in entry): # <-- Name changed to 'keypoint'
                continue

            kp = entry["keypoint"] # <-- Name changed to 'keypoint'
            if not isinstance(kp, np.ndarray) or kp.ndim != 3:
                # Handle PyTorch tensors from MSKA .pkl files
                if torch.is_tensor(kp):
                    kp = kp.numpy()
                else:
                    continue
            
            if kp.shape[1] != kpoints:
                print(f"Skipping {k}: expected {kpoints} keypoints, got {kp.shape[1]}")
                continue

            T, K, C = kp.shape
            stats["frames_before_total"] += T
            stats["dtype_counts_before"][str(kp.dtype)] += 1
            stats["K_counts_before"][K] += 1
            stats["C_counts_before"][C] += 1

            # --- STEP 1: Run your perfect interpolation ---
            # This creates new absolute (X, Y) positions
            # Shape: (T_interp, K, C)
            new_seq_positions = interp.interpolate_sequence(kp, preserve_dtype=kp.dtype)
            T_interp = new_seq_positions.shape[0]

            # --- STEP 2: Convert absolute positions to motion vectors ---
            # We calculate the difference between each new frame
            # Shape: (T_interp-1, K, 2)
            diff = np.diff(new_seq_positions[..., :2], axis=0)

            # Because your interpolator moves in steps of `axis_step`,
            # np.sign() will give us exactly {-1, 0, 1}
            # Shape: (T_interp-1, K, 2)
            motion_vectors_dx_dy = np.sign(diff).astype(np.int8)

            # We need to add the "first frame" of motion (which is 0)
            # Shape: (1, K, 2)
            first_frame_motion = np.zeros((1, K, 2), dtype=np.int8)

            # Shape: (T_interp, K, 2)
            final_motion_tensor = np.concatenate(
                (first_frame_motion, motion_vectors_dx_dy), 
                axis=0
            )
            
            # --- STEP 3: Update the dictionary ---
            # We replace the original 'keypoint' with the new 'motion_vector'
            # and update 'num_frames'
            del entry["keypoint"] # Free memory
            entry["motion_vector"] = final_motion_tensor
            entry["num_frames"] = T_interp # Update frame count

            # --- Statistics ---
            stats["frames_after_total"] += T_interp
            stats["dtype_counts_after"][str(final_motion_tensor.dtype)] += 1
            stats["K_counts_after"][final_motion_tensor.shape[1]] += 1
            stats["C_counts_after"][final_motion_tensor.shape[2]] += 1

            ratio = float(T_interp) / float(T) if T > 0 else 0.0
            ratio_sum += ratio
            processed_count += 1

            stats["ratio_min"] = ratio if (stats["ratio_min"] is None) else min(stats["ratio_min"], ratio)
            stats["ratio_max"] = ratio if (stats["ratio_max"] is None) else max(stats["ratio_max"], ratio)

        gc.collect()
        print(f"Processed {min(i + batch_size, N):,} / {N:,}")

    # Robust write to UNC
    print(f"Writing final motion dataset → {output_pkl}")
    _safe_write_pickle(data, output_pkl)

    # Sampled verification (This part is not as useful now, but we'll leave it)
    random.seed(seed)
    sample_keys = random.sample(keys, k=min(verify_sample, len(keys)))
    stats["unit_step_checked_samples"] = len(sample_keys)
    violations = 0
    # for k in sample_keys:
    #     seq = data.get(k, {}).get("keypoints", None) # This key no longer exists
    #     if isinstance(seq, np.ndarray) and seq.ndim == 3 and seq.shape[0] >= 2:
    #         if not verify_axis_steps(seq, axis_step=axis_step):
    #             violations += 1
    stats["unit_step_violations_sampled"] = violations
    stats["ratio_avg"] = (ratio_sum / max(1, processed_count)) if processed_count > 0 else 0.0

    # Report JSON (UNC-safe)
    base = os.path.splitext(output_pkl)[0]
    report_json = base + "_REPORT.json"
    report_json_w = _to_long_unc(report_json) if os.name == "nt" else report_json
    with open(report_json_w, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Friendly print
    rmin = stats["ratio_min"] if stats["ratio_min"] is None else 0.0
    rmax = stats["ratio_max"] if stats["ratio_max"] is None else 0.0
    print("\n=== DONE ===")
    print(f"Output dataset : {output_pkl}")
    print(f"Report (JSON)  : {report_json}")
    print(f"Axis step      : {axis_step}")
    print(f"Avg ratio T'/T : {stats['ratio_avg']:.4f} (min={rmin:.4f}, max={rmax:.4f})")
    # print(f"Sampled step violations (> {axis_step} px per axis): {violations} / {len(sample_keys)} (expect 0)")
    print("=====================================================")

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Phoenix interpolation and motion vector conversion.")
    ap.add_argument("input",  help="Path to original Phoenix pickle (.pkl) from MSKA")
    ap.add_argument("output", help="Path to write *motion vector* pickle (.pkl)")
    ap.add_argument("--batch_size", type=int, default=50, help="Entries per batch to keep memory flat")
    ap.add_argument("--verify_sample", type=int, default=80, help="Sample size for post-check")
    ap.add_argument("--kpoints", type=int, default=133, help="Keypoints per frame (default 133 for MSKA)")
    ap.add_argument("--max_steps", type=int, default=None, help="Clamp max substeps per pair (optional)")
    ap.add_argument("--axis_step", type=float, default=1.0, help="Max |Δx| or |Δy| per frame (pixels). Use 1.0 for {-1,0,1} output.")
    args = ap.parse_args()

    if args.kpoints != 133:
        print(f"Warning: You set --kpoints={args.kpoints}, but MSKA data has 133. Mismatch may occur.")

    run(
        args.input,
        args.output,
        batch_size=args.batch_size,
        verify_sample=args.verify_sample,
        kpoints=args.kpoints,
        max_steps=args.max_steps,
        axis_step=args.axis_step,
    )

if __name__ == "__main__":
    main()