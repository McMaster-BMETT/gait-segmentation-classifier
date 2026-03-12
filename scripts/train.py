"""Train the gait segmentation model."""

import argparse
import os
import glob
import numpy as np
import yaml
import torch

from gait_classifier.data.dataset import create_dataloaders
from gait_classifier.data.preprocessing import Normalizer
from gait_classifier.models.gait_tcn import GaitTCN
from gait_classifier.training.losses import WeightedCELoss, compute_class_weights
from gait_classifier.training.trainer import GaitTrainer
from gait_classifier.training.metrics import evaluate_model


def load_sessions(data_dir: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load all .npz sessions from data directory."""
    sessions = []
    for npz_path in sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)):
        f = np.load(npz_path)
        sessions.append((f["data"], f["labels"]))
    return sessions


def split_sessions(sessions, train_ratio, val_ratio, rng):
    """Stratified split by healing stage (sessions are already grouped by stage)."""
    rng.shuffle(sessions)
    n = len(sessions)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return sessions[:n_train], sessions[n_train:n_train + n_val], sessions[n_train + n_val:]


def main():
    parser = argparse.ArgumentParser(description="Train gait segmentation model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--save-path", default="model.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = args.data_dir or config["data"]["output_dir"]
    print(f"Loading data from {data_dir}...")
    sessions = load_sessions(data_dir)
    print(f"Loaded {len(sessions)} sessions")

    # Split
    train_sessions, val_sessions, test_sessions = split_sessions(
        sessions, config["data"]["train_ratio"], config["data"]["val_ratio"], rng
    )
    print(f"Split: {len(train_sessions)} train, {len(val_sessions)} val, {len(test_sessions)} test")

    # Normalize using training data stats
    normalizer = Normalizer()
    train_concat = np.concatenate([s[0] for s in train_sessions], axis=1)
    normalizer.fit(train_concat)

    train_sessions = [(normalizer.transform(d), l) for d, l in train_sessions]
    val_sessions = [(normalizer.transform(d), l) for d, l in val_sessions]
    test_sessions = [(normalizer.transform(d), l) for d, l in test_sessions]

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_sessions, val_sessions, test_sessions,
        batch_size=config["training"]["batch_size"],
    )

    # Model
    model_cfg = config["model"]
    model = GaitTCN(
        num_channels=model_cfg["num_channels"],
        hidden_channels=model_cfg["hidden_channels"],
        num_classes=model_cfg["num_classes"],
        num_blocks=model_cfg["num_blocks"],
        kernel_size=model_cfg["kernel_size"],
        dilations=model_cfg["dilations"],
        dropout=model_cfg["dropout"],
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss with class weights
    train_labels = [torch.from_numpy(l) for _, l in train_sessions]
    weights = compute_class_weights(train_labels, num_classes=model_cfg["num_classes"])
    print(f"Class weights: {weights.tolist()}")
    criterion = WeightedCELoss(
        weight=weights,
        label_smoothing=config["training"]["label_smoothing"],
    )

    # Train
    trainer = GaitTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        epochs=config["training"]["epochs"],
        patience=config["training"]["early_stopping_patience"],
        device=args.device,
        augmentation=config.get("augmentation"),
    )

    print(f"\nTraining on {args.device}...")
    result = trainer.train()
    print(f"\nTraining complete. Best val loss: {result['best_val_loss']:.4f} "
          f"after {result['epochs_trained']} epochs")

    # Evaluate on test set
    print("\nTest set evaluation:")
    test_metrics = evaluate_model(model, test_loader, device=args.device)
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    for phase, metrics in test_metrics["per_phase_f1"].items():
        print(f"  {phase}: F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}")

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model_cfg,
        "normalizer_params": normalizer.get_params(),
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
    }, args.save_path)
    print(f"\nModel saved to {args.save_path}")


if __name__ == "__main__":
    main()
