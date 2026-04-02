"""Export pre-computed demo data for the Dash dashboard.

Run once before the expo to generate demo/demo_data.json.
Loads the trained model, runs inference on synthetic data for all 4 healing
stages, computes clinical metrics, and serialises everything to JSON.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gait_classifier.constants import HEALING_STAGES, PHASE_NAMES, CHANNEL_NAMES
from gait_classifier.data.preprocessing import Normalizer
from gait_classifier.models.gait_tcn import GaitTCN
from gait_classifier.clinical.quad_inhibition import compute_qii
from gait_classifier.clinical.hq_ratio import compute_hq_conventional, compute_hq_functional
from gait_classifier.clinical.phase_rom import compute_phase_rom
from gait_classifier.clinical.limb_symmetry import compute_lsi_temporal


def _sanitize(val):
    """Convert numpy/nan values to JSON-safe Python types."""
    if isinstance(val, (np.floating, float)):
        if np.isnan(val) or np.isinf(val):
            return None
        return round(float(val), 4)
    if isinstance(val, (np.integer, int)):
        return int(val)
    return val


def main():
    parser = argparse.ArgumentParser(description="Export demo data for dashboard")
    parser.add_argument("--model-path", default="model.pth")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--output", default="demo/demo_data.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    model_cfg = checkpoint["model_config"]
    model = GaitTCN(**{k: v for k, v in model_cfg.items() if k != "type"})
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    # Restore normalizer
    normalizer = Normalizer()
    params = checkpoint["normalizer_params"]
    normalizer.mean = np.array(params["mean"]).reshape(-1, 1)
    normalizer.std = np.array(params["std"]).reshape(-1, 1)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    test_metrics = checkpoint.get("test_metrics", {})

    demo = {
        "stages": {},
        "phase_names": PHASE_NAMES,
        "channel_names": CHANNEL_NAMES,
        "model_info": {
            "total_params": total_params,
            "accuracy": _sanitize(test_metrics.get("accuracy")),
            "num_blocks": model_cfg.get("num_blocks", 4),
            "hidden_channels": model_cfg.get("hidden_channels", 64),
            "kernel_size": model_cfg.get("kernel_size", 3),
            "dilations": model_cfg.get("dilations", [1, 2, 4, 8]),
        },
    }

    for stage_name in HEALING_STAGES:
        stage_dir = os.path.join(args.data_dir, stage_name)
        npz_files = sorted(glob.glob(os.path.join(stage_dir, "*.npz")))
        if not npz_files:
            print(f"  [SKIP] No data for {stage_name}")
            continue

        f = np.load(npz_files[0])
        raw_data = f["data"]      # (8, T)
        true_labels = f["labels"]  # (T,)
        norm_data = normalizer.transform(raw_data)

        # Predict phases
        with torch.no_grad():
            x = torch.from_numpy(norm_data).float().unsqueeze(0).to(args.device)
            logits = model(x)
            pred_labels = logits.argmax(dim=1).squeeze().cpu().numpy()

        # Extract raw channels
        imu_yaw = raw_data[2]
        emg_quad = raw_data[6]
        emg_ham = raw_data[7]

        # Clinical metrics
        qii = compute_qii(emg_quad, pred_labels)
        hq_func = compute_hq_functional(emg_quad, emg_ham, pred_labels)
        rom = compute_phase_rom(imu_yaw, pred_labels)
        lsi = compute_lsi_temporal(pred_labels)

        # Knee flexion angle (offset from baseline)
        knee_flexion = (imu_yaw - 180.0).tolist()

        # Trim to ~3-4 gait cycles for display (approx 200-300 samples at 50Hz)
        display_len = min(len(knee_flexion), 300)

        demo["stages"][stage_name] = {
            "knee_flexion": knee_flexion[:display_len],
            "emg_quad": emg_quad[:display_len].tolist(),
            "emg_ham": emg_ham[:display_len].tolist(),
            "true_labels": true_labels[:display_len].tolist(),
            "pred_labels": pred_labels[:display_len].tolist(),
            "metrics": {
                "qii": _sanitize(qii),
                "hq_functional": _sanitize(hq_func),
                "swing_peak_flexion": _sanitize(rom.get("swing_peak_flexion")),
                "lsi_temporal": _sanitize(lsi),
            },
        }
        print(f"  [OK] {stage_name}: {display_len} samples, QII={_sanitize(qii)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(demo, fp, indent=2)
    print(f"\nExported to {args.output}")


if __name__ == "__main__":
    main()
