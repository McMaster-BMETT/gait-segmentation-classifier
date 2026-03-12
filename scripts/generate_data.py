"""Generate synthetic gait data for all healing stages."""

import argparse
import os
import numpy as np
import yaml

from gait_classifier.data.healing_profiles import ALL_PROFILES
from gait_classifier.data.synthetic import GaitCycleGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic gait data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cycles-per-stage", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir or config["data"]["output_dir"]
    cycles_per_stage = args.cycles_per_stage or config["data"]["cycles_per_stage"]
    sampling_rate = config["sensor"]["sampling_rate"]

    rng = np.random.default_rng(args.seed)
    total_samples = 0

    for profile in ALL_PROFILES:
        stage_dir = os.path.join(output_dir, profile.name)
        os.makedirs(stage_dir, exist_ok=True)

        gen = GaitCycleGenerator(profile, sampling_rate=sampling_rate, rng=rng)

        # Generate individual cycles and save as sessions of 10 cycles each
        cycles_per_session = 10
        n_sessions = cycles_per_stage // cycles_per_session

        for session_idx in range(n_sessions):
            data, labels = gen.generate_session(num_cycles=cycles_per_session)
            np.savez(
                os.path.join(stage_dir, f"session_{session_idx:03d}.npz"),
                data=data,
                labels=labels,
                stage_id=profile.stage_id,
                stage_name=profile.name,
            )
            total_samples += data.shape[1]

        print(f"{profile.name}: {n_sessions} sessions, "
              f"{cycles_per_stage} cycles")

    print(f"\nTotal timesteps: {total_samples}")
    print(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
