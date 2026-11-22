"""Evaluation geometry, locked comparisons, and isolation of the independent spatial probe."""

import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch

from scripts.analysis.visual_v4_compare import compare, parent_bootstrap
from scripts.analysis.visual_v4_eval import box_overlap, evaluate, metric_point, paired_rows, peak3
from scripts.analysis.visual_v4_probe import MapBoxProbe, check_partitions, fit_probe


class DummyPolicy:
    model = SimpleNamespace(config=SimpleNamespace(image_size=32, attention_layers=(0, 1)))
    tokenizer = staticmethod(lambda task: (np.array([1, 2, 0]), np.array([True, True, False])))

    def infer(self, images, task):
        weights = torch.full((1, 16), 0.001)
        weights[0, 0 if task == "milk" else 15] = 0.985
        layers = weights[:, None, :].expand(-1, 2, -1)
        return dict(
            current_map=weights,
            subject_map=weights,
            layer_maps=layers,
            alpha=torch.tensor([[0.2, 0.8]]),
            text_mask=torch.tensor([[True, True]]),
            visual_attention_mass=torch.full((1, 2, 2, 2), 0.5),
        )


def test_geometry_and_cluster_bootstrap():
    values = np.zeros((4, 4))
    values[3, 2] = 1.0
    np.testing.assert_allclose(peak3(values.ravel()), [0.625, 0.875])
    coverage = box_overlap([0, 0, 0.125, 0.25], 4)
    np.testing.assert_allclose(coverage[:2], [0.5, 0])
    np.testing.assert_allclose(metric_point(np.array([0.5, 0.5]), (40, 90, 3), 224, "source"), [45, 20])
    result = parent_bootstrap([("a", 1), ("a", 1), ("b", 0)], 100, 42)
    assert result["n"] == 3 and result["parents"] == 2 and result["delta"] == pytest.approx(2 / 3)
    assert result["ci95"] == [0.0, 1.0]


def test_eval_same_input_probe_and_pairs(tmp_path):
    Image.new("RGB", (32, 32), "gray").save(tmp_path / "obs.png")
    base = dict(
        scene="one",
        image="obs.png",
        boxes=dict(milk=[0, 0, 8, 8], cup=[24, 24, 32, 32]),
        subject_box=[0, 0, 8, 8],
    )
    rows = []
    for split in ["Fit", "Check", "D"]:
        for t in ("milk", "cup"):
            rows.append(dict(base, split=split, parent=split, id=split + t, task=t, target=t, pair_id="pair"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(protocol="synthetic-regression", metric_space="source", rows=rows)))
    out = tmp_path / "eval"
    report = evaluate(DummyPolicy(), manifest, out, panels=1)
    assert report["groups"]["all"]["current"]["hit10"] == {"mean": 1.0, "n": 6}
    assert report["groups"]["split_scene/D/one"]["layers"]["1"]["visual_mass"]["mean"] == pytest.approx(0.5)
    assert report["pair_summary"]["pair_id/D"]["both_hit10"]["mean"] == 1.0
    assert (out / "panels/00000.jpg").exists()
    assert report["rows"][0]["alpha"]["pieces"] == ["1", "2"]
    assert (
        compare({"baseline": report, "copy": report}, "baseline", draws=10)["copy"]["all"]["current.hit10"][
            "delta"
        ]
        == 0
    )
    with pytest.raises(ValueError, match="identical"):
        compare({"baseline": report, "copy": dict(report, input_sha256="no")}, "baseline")
    paired = [dict(x) for x in report["rows"][:2]]
    paired[1]["rgb_hash"] = "not-identical"
    with pytest.raises(ValueError, match="exact pixels"):
        paired_rows(paired)
    with pytest.raises(ValueError, match="share parent"):
        check_partitions([dict(split="Fit", parent="a"), dict(split="Check", parent="a")])
    probe = MapBoxProbe()
    x = torch.ones((1, 16), requires_grad=True)
    probe.loss(x, torch.tensor([[0.0, 0.0, 0.4, 0.4]])).backward()
    assert x.grad is None
    assert any(p.grad is not None for p in probe.parameters())
    result = fit_probe(out, tmp_path / "probe", steps=2, batch=2, seed=42)
    assert result["fit_samples"] == result["check_samples"] == 2
    assert result["by_split"]["D"]["iou"]["n"] == 2


def test_absent_targets_and_moves():
    a = dict(
        split="D",
        parent="traj",
        scene="host1",
        move_id="across",
        id="one",
        task="milk",
        target="milk",
        current=dict(hit10=1, peak3_xy=[20, 20], true_center_xy=[20, 20]),
    )
    b = dict(a, id="two", current=dict(hit10=1, peak3_xy=[25, 22], true_center_xy=[27, 22]))
    pairs = paired_rows([a, b])
    assert pairs[0]["displacement_error_px"] == 2.0
    b["current"] = dict(peak3_xy=[25, 22])
    assert "displacement_error_px" not in paired_rows([a, b])[0]


def test_history_bbox_uses_regressed_center_and_keeps_native_outputs_separate(tmp_path):
    class LegacyPolicy:
        model = SimpleNamespace(config=SimpleNamespace(image_size=32, patch_size=8, attention_layers=(0, 1)))
        precision = "float32"

        def infer(self, images, task):
            return {
                "bbox_xyxy": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "subject_bbox_xyxy": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            }

    Image.new("RGB", (32, 32), "black").save(tmp_path / "legacy.png")
    (tmp_path / "legacy.json").write_text(
        json.dumps(
            dict(
                protocol="technically_matched",
                metric_space="source",
                rows=[
                    dict(
                        id="one",
                        scene="s",
                        split="Check",
                        parent="p",
                        image="legacy.png",
                        task="target",
                        boxes={"target": [0, 0, 32, 32]},
                        subject_box=[0, 0, 32, 32],
                        target="target",
                    )
                ],
            )
        )
    )
    report = evaluate(LegacyPolicy(), tmp_path / "legacy.json", tmp_path / "out", panels=1)
    assert report["readout"] == "regressed_bbox"
    row = report["rows"][0]
    assert row["current"]["predicted_center_xy"] == [16.0, 16.0] and row["current"]["center_error_px"] == 0
    assert row["layers"] == {} and row["alpha_stats"] == {} and "visual_mass" not in row["current"]
