"""Evaluation geometry, locked comparisons, and isolation of the independent spatial probe."""

import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch

from scripts.analysis.visual_primitive_compare import compare, parent_bootstrap
from openpi.models.visual_comparisons import VisualPrimitive
from openpi.models.visual_primitive import patch_coverage
from openpi.transforms import normalized_image_boxes, resize_image, source_pixel_geometry
from scripts.analysis.visual_primitive_eval import box_overlap, evaluate, map_score, metric_point, paired_rows, peak3
from scripts.analysis.visual_primitive_probe import MapBoxProbe, check_partitions, fit_probe


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


def test_training_and_eval_share_peak3_bbox_center_and_source_pixels():
    shape, size, side = (128, 256), 224, 16
    weights = torch.zeros(side, side)
    weights[0, 0] = .25
    weights[8:11, 10:13] = .75 / 9
    weights = weights.flatten()[None]
    # Current center is on the cluster, Subject center is 11 source pixels below it.
    current_box, subject_box = normalized_image_boxes([[145, 60, 223, 116], [145, 71, 223, 127]], shape, size)
    # Only the left edge is visible, so the mask centroid differs from both box centers.
    visible = np.zeros(shape, dtype=np.float32)
    visible[74:84, 147:159] = 1
    coverage = patch_coverage(torch.from_numpy(resize_image(visible, size))[None], side)
    pixel_scale, _ = source_pixel_geometry(shape, size)
    targets = dict(current_box=torch.from_numpy(current_box[None]), subject_box=torch.from_numpy(subject_box[None]),
                   center_px_scale=torch.from_numpy(pixel_scale[None].astype(np.float32)),
                   entity_boxes=torch.zeros(1, 1, 4), entity_valid=torch.zeros(1, 1, dtype=torch.bool),
                   positive_entity=torch.tensor([-1]))
    pred = dict(current_map=weights, subject_map=weights, layer_maps=weights[:, None],
                visual_attention_mass=torch.ones(1, 1, 1, 1), alpha=torch.ones(1, 1),
                text_mask=torch.ones(1, 1, dtype=torch.bool))
    diagnostics = {}
    valid = torch.tensor([True])
    model = SimpleNamespace(config=SimpleNamespace(attention_layers=(0,)))
    VisualPrimitive.map_statistics(model, diagnostics, pred, targets, coverage, coverage, torch.zeros(1, 1), valid, valid)
    for role, box in [('current', current_box), ('subject', subject_box)]:
        score = map_score(weights[0].numpy(), box, coverage[0].numpy(), [], shape, size, 'source')
        for metric in ('peak3_error_px', 'peak3_hit10', 'argmax_error_px', 'argmax_hit10'):
            torch_value, count = diagnostics[f'grounding_{role}_{metric}'].tolist()
            assert count == 1 and torch_value == pytest.approx(score[metric], abs=1e-5)
        assert score['peak3_hit10'] == (role == 'current')
        assert score['argmax_hit10'] == 0.
        assert score['roi_mass'] == pytest.approx(diagnostics[f'grounding_{role}_gt_mass'][0].item())
        if role == 'current':
            assert score['kl'] == pytest.approx(diagnostics['grounding_current_map_kl'][0].item())
    assert map_score(weights[0].numpy(), subject_box, coverage[0].numpy(), [], shape, size, 'model')['peak3_hit10'] == 1.
    override = map_score(weights[0].numpy(), subject_box, coverage[0].numpy(), [], shape, size, 'source',
                         gt_center=(current_box[:2] + current_box[2:]) * .5)
    assert override['peak3_hit10'] == 1.
    mask_only = map_score(weights[0].numpy(), None, coverage[0].numpy(), [], shape, size, 'source')
    assert 'roi_mass' in mask_only and 'peak3_hit10' not in mask_only


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
    assert report["groups"]["all"]["current"]["peak3_hit10"] == {"mean": 1.0, "n": 6}
    assert report["groups"]["split_scene/D/one"]["layers"]["1"]["visual_mass"]["mean"] == pytest.approx(0.5)
    assert report["pair_summary"]["pair_id/D"]["both_hit10"]["mean"] == 1.0
    assert report["pair_summary"]["pair_id/D"]["both_peak3_hit10"]["mean"] == 1.0
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
