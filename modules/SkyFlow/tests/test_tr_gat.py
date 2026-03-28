"""Tests for TR-GAT architecture."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from skyflow.models.tr_gat import TRGAT, TRGATLayer
from skyflow.models.temporal_encoding import SinusoidalTemporalEncoding
from skyflow.models.conflict_head import ConflictScoringHead
from skyflow.models.resolution import ResolutionModule


class TestTemporalEncoding:
    def test_output_shape(self):
        enc = SinusoidalTemporalEncoding(d_phi=32)
        delta = torch.tensor([0.0, 1.0, 5.0, 10.0])
        out = enc(delta)
        assert out.shape == (4, 32)

    def test_zero_delta(self):
        enc = SinusoidalTemporalEncoding(d_phi=32)
        out = enc(torch.tensor([0.0]))
        assert out.shape == (1, 32)
        assert torch.isfinite(out).all()

    def test_different_deltas_different_encodings(self):
        enc = SinusoidalTemporalEncoding(d_phi=32)
        out1 = enc(torch.tensor([1.0]))
        out2 = enc(torch.tensor([10.0]))
        assert not torch.allclose(out1, out2)


class TestTRGATLayer:
    def test_forward_shape(self):
        layer = TRGATLayer(in_dim=128, out_dim=128, num_heads=4, num_relations=6, temporal_dim=32)
        x = torch.randn(100, 128)
        edge_indices = {0: torch.stack([torch.randint(0, 100, (50,)), torch.randint(0, 100, (50,))])}
        temporal_enc = {0: torch.randn(50, 32)}
        out = layer(x, edge_indices, temporal_enc)
        assert out.shape == (100, 128)

    def test_empty_edges(self):
        layer = TRGATLayer(in_dim=128, out_dim=128, num_heads=4, num_relations=6, temporal_dim=32)
        x = torch.randn(10, 128)
        out = layer(x, {}, {})
        assert out.shape == (10, 128)


class TestTRGAT:
    def test_forward(self):
        model = TRGAT(node_feature_dim=23, embed_dim=128, num_layers=4,
                      num_heads=4, num_relations=6, temporal_dim=32, recurrent_dim=64)
        feats = torch.randn(50, 23)
        ei = {0: torch.stack([torch.randint(0, 50, (30,)), torch.randint(0, 50, (30,))])}
        ed = {0: torch.rand(30)}
        emb, state = model(feats, ei, ed)
        assert emb.shape == (50, 128)
        assert state.shape == (50, 64)

    def test_with_recurrent_state(self):
        model = TRGAT(node_feature_dim=23, embed_dim=128, num_layers=2,
                      num_heads=4, num_relations=6, temporal_dim=32, recurrent_dim=64)
        feats = torch.randn(20, 23)
        ei = {0: torch.stack([torch.randint(0, 20, (10,)), torch.randint(0, 20, (10,))])}
        ed = {0: torch.rand(10)}
        prev_state = torch.randn(20, 64)
        emb, state = model(feats, ei, ed, recurrent_state=prev_state)
        assert emb.shape == (20, 128)
        assert state.shape == (20, 64)

    def test_parameter_count(self):
        model = TRGAT(node_feature_dim=23, embed_dim=128, num_layers=4,
                      num_heads=4, num_relations=6, temporal_dim=32, recurrent_dim=64)
        assert model.count_parameters() > 0


class TestConflictHead:
    def test_forward(self):
        head = ConflictScoringHead(embed_dim=128, recurrent_dim=64)
        P = 20
        h_i = torch.randn(P, 128)
        h_j = torch.randn(P, 128)
        s_i = torch.randn(P, 64)
        s_j = torch.randn(P, 64)
        out = head(h_i, h_j, s_i, s_j)
        assert out.shape == (P,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_with_edge_features(self):
        head = ConflictScoringHead(embed_dim=128, recurrent_dim=64, edge_feature_dim=16)
        P = 10
        out = head(
            torch.randn(P, 128), torch.randn(P, 128),
            torch.randn(P, 64), torch.randn(P, 64),
            edge_feat=torch.randn(P, 16),
        )
        assert out.shape == (P,)


class TestResolution:
    def test_forward(self):
        res = ResolutionModule(embed_dim=128, pgd_steps=5)
        emb = torch.randn(4, 128)
        pos = torch.randn(4, 3) * 100
        vel = torch.randn(4, 3) * 10
        offsets, info = res(emb, pos, vel)
        assert offsets.shape == (4, 3)
        assert "steps" in info

    def test_single_uav(self):
        res = ResolutionModule(embed_dim=128)
        offsets, info = res(torch.randn(1, 128), torch.randn(1, 3), torch.randn(1, 3))
        assert offsets.shape == (1, 3)
        assert info["steps"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
