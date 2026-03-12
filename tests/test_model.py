"""Tests for the GaitTCN model."""

import torch
import pytest

from gait_classifier.models.gait_tcn import GaitTCN, TCNBlock


class TestTCNBlock:
    def test_output_shape_same_channels(self):
        block = TCNBlock(64, 64, kernel_size=3, dilation=2)
        x = torch.randn(4, 64, 100)
        out = block(x)
        assert out.shape == x.shape

    def test_output_shape_different_channels(self):
        block = TCNBlock(8, 64, kernel_size=3, dilation=1)
        x = torch.randn(4, 8, 100)
        out = block(x)
        assert out.shape == (4, 64, 100)


class TestGaitTCN:
    def test_output_shape(self):
        model = GaitTCN(num_channels=8, hidden_channels=64, num_classes=6)
        x = torch.randn(4, 8, 200)
        out = model(x)
        assert out.shape == (4, 6, 200)

    def test_variable_length(self):
        model = GaitTCN()
        for T in [50, 100, 500]:
            x = torch.randn(2, 8, T)
            out = model(x)
            assert out.shape == (2, 6, T)

    def test_single_sample(self):
        model = GaitTCN()
        x = torch.randn(1, 8, 30)
        out = model(x)
        assert out.shape == (1, 6, 30)

    def test_overfit_single_batch(self):
        """Model should be able to overfit a single batch."""
        torch.manual_seed(42)
        model = GaitTCN(num_channels=8, hidden_channels=64, num_classes=6,
                        num_blocks=2, dilations=[1, 2], dropout=0.0)
        x = torch.randn(4, 8, 50)
        labels = torch.randint(0, 6, (4, 50))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for _ in range(300):
            logits = model(x)
            # cross_entropy accepts (B, C, T) logits and (B, T) labels directly
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(x).argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()

        assert accuracy > 0.90, f"Failed to overfit: accuracy={accuracy:.3f}"
