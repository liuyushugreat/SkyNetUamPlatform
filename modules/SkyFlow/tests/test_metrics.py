"""Tests for conflict detection metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from skyflow.training.metrics import ConflictMetrics
from skyflow.training.losses import FocalLoss


class TestConflictMetrics:
    def test_perfect_prediction(self):
        m = ConflictMetrics(threshold=0.5)
        preds = torch.tensor([0.9, 0.8, 0.1, 0.05])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        m.update(preds, labels)
        result = m.compute()
        assert result.cdr == 1.0
        assert result.far == 0.0
        assert result.f1 == 1.0

    def test_all_wrong(self):
        m = ConflictMetrics(threshold=0.5)
        preds = torch.tensor([0.1, 0.1, 0.9, 0.9])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        m.update(preds, labels)
        result = m.compute()
        assert result.cdr == 0.0

    def test_reset(self):
        m = ConflictMetrics()
        m.update(torch.tensor([0.5]), torch.tensor([1.0]))
        m.reset()
        assert len(m.all_preds) == 0

    def test_multi_batch(self):
        m = ConflictMetrics(threshold=0.5)
        m.update(torch.tensor([0.9, 0.1]), torch.tensor([1.0, 0.0]))
        m.update(torch.tensor([0.8, 0.2]), torch.tensor([1.0, 0.0]))
        result = m.compute()
        assert result.num_pairs == 4
        assert result.cdr == 1.0


class TestFocalLoss:
    def test_output_scalar(self):
        loss = FocalLoss(gamma=2.0)
        pred = torch.tensor([0.9, 0.1, 0.5])
        target = torch.tensor([1.0, 0.0, 1.0])
        out = loss(pred, target)
        assert out.dim() == 0
        assert out.item() > 0

    def test_perfect_pred_low_loss(self):
        loss = FocalLoss(gamma=2.0)
        perfect = loss(torch.tensor([0.99, 0.01]), torch.tensor([1.0, 0.0]))
        imperfect = loss(torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0]))
        assert perfect.item() < imperfect.item()

    def test_gamma_effect(self):
        pred = torch.tensor([0.7, 0.3])
        target = torch.tensor([1.0, 0.0])
        loss_g0 = FocalLoss(gamma=0.0)(pred, target)
        loss_g2 = FocalLoss(gamma=2.0)(pred, target)
        assert loss_g2.item() < loss_g0.item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
