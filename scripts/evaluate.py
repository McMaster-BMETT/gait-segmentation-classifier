"""Evaluate a trained model and compute clinical metrics on test data."""

import argparse
import os
import glob
import numpy as np
import yaml
import torch

from gait_classifier.data.preprocessing import Normalizer
from gait_classifier.models.gait_tcn import GaitTCN
from gait_classifier.clinical.quad_inhibition import compute_qii
from gait_classifier.clinical.hq_ratio import compute_hq_conventional, compute_hq_functional
from gait_classifier.clinical.phase_rom import compute_phase_rom
from gait_classifier.clinical.limb_symmetry import compute_lsi_temporal
from gait_classifier.constants import HEALING_STAGES


def main():
    parser = argparse.ArgumentParser(description="Evaluate model and compute clinical metrics")
    parser.add_argument("--model-path", default="model.pth")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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

    print("Clinical Metrics by Healing Stage")
    print("=" * 60)

    for stage_name in HEALING_STAGES:
        stage_dir = os.path.join(args.data_dir, stage_name)
        npz_files = sorted(glob.glob(os.path.join(stage_dir, "*.npz")))
        if not npz_files:
            continue

        # Load first session for demo
        f = np.load(npz_files[0])
        raw_data = f["data"]  # (8, T) - keep raw for clinical metrics
        norm_data = normalizer.transform(raw_data)

        # Predict phases
        with torch.no_grad():
            x = torch.from_numpy(norm_data).float().unsqueeze(0).to(args.device)
            logits = model(x)
            pred_labels = logits.argmax(dim=1).squeeze().cpu().numpy()

        # Extract raw channels
        imu_yaw = raw_data[2]     # flexion axis
        emg_quad = raw_data[6]    # quad
        emg_ham = raw_data[7]     # hamstring

        # Compute clinical metrics
        qii = compute_qii(emg_quad, pred_labels)
        hq_conv = compute_hq_conventional(emg_quad, emg_ham, pred_labels)
        hq_func = compute_hq_functional(emg_quad, emg_ham, pred_labels)
        rom = compute_phase_rom(imu_yaw, pred_labels)
        lsi = compute_lsi_temporal(pred_labels)

        print(f"\n{stage_name}")
        print(f"  QII:                  {qii:.3f}")
        print(f"  H:Q Conventional:     {hq_conv:.3f}")
        print(f"  H:Q Functional:       {hq_func:.3f}")
        print(f"  Swing Peak Flexion:   {rom['swing_peak_flexion']:.1f}°")
        print(f"  Swing Clearance:      {rom['swing_clearance']:.2f}")
        print(f"  Stance Flexion Range: {rom['stance_flexion_range']:.1f}°")
        print(f"  LSI (temporal):       {lsi:.1f}%")


if __name__ == "__main__":
    main()
