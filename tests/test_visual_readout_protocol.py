import json
import os
from types import SimpleNamespace

import pytest

from experiments.visual_primitive.run import native


def test_native_relocation_preserves_head_selection_and_rejects_changed_query(tmp_path):
    from eskill_bench.protocols.assets import NATIVE, Settings, digest, read

    bundle = os.environ.get("ESKILL_EVAL_BUNDLE")
    if not bundle:
        pytest.skip("Set ESKILL_EVAL_BUNDLE to the sealed extraction")
    cfg = tmp_path / "host.json"
    cfg.write_text(
        json.dumps(
            dict(
                paths=dict(bundle=bundle, base="base", tokenizer="tokenizer", output="runs"),
                device="cpu",
                path_map={"/data/home": "/other/host"},
            )
        )
    )
    settings = Settings(cfg)
    recipe = tmp_path / "config.json"
    recipe.write_text("{}")
    run = tmp_path / "run"
    native(settings, run, SimpleNamespace(phase="prepare", recipe=recipe))
    native(settings, run, SimpleNamespace(phase="use-lock"))
    golden = read(settings.bundle.file(NATIVE + "selection.json"))
    assert read(run / "selection.json") == dict(golden, dev_manifest_sha256=digest(run / "dev_rows.json"))
    rows = read(run / "test_rows.json")
    rows[0]["prompt"] = "different"
    (run / "test_rows.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="test differs"):
        native(settings, run, SimpleNamespace(phase="score"))
