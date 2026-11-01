# Code provenance

The π0.5 architecture, adaptive RMS normalization, tokenizer prompt format and
flow-matching conventions follow [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi),
licensed under Apache-2.0. The local reference used during integration is
RoboTwin/XPolicyLab/policy/Pi_05/openpi; AtomicVLA and GuidedVLA informed the
directory organization. ESkill's event-conditioned branches are implemented in
this repository. Hugging Face Transformers provides the PaliGemma, SigLIP and
Gemma parameter modules; their installed source is not modified.

The OpenPI Apache-2.0 license is reproduced in licenses/openpi-LICENSE.

The LIBERO evaluation observation/action conventions follow the official OpenPI
client at commit 215abfb217dbac7d5f1273282331b9b1866c0479. The baseline launcher
uses that separate official checkout and its pinned LeRobot dependency; source,
asset and dependency revisions are recorded in
`../workspace/envs/libero-evaluation/manifest.json`.

The pure π0.5 SFT path also follows that pinned checkout's LIBERO configuration,
quantile normalization, end-of-episode action padding, and image augmentation
conventions. The local checkpoint service reuses its WebSocket server module
without importing the official model namespace. LoRA, batch size and training
schedule used in each experiment are recorded separately from the upstream recipe.
