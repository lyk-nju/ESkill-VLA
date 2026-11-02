"""SAM3 explicit-instance tracker; uses local weights, no concept-wide selection."""
from pathlib import Path

import numpy as np


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1] if len(xs) else None


class SamDetector:
    """SAM3's native text detector supplies masks before instance selection."""
    def __init__(self, config):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.config = config
        torch.manual_seed(config.get("seed", 42))
        self.model = build_sam3_image_model(checkpoint_path=config["checkpoint_path"],
                                          load_from_HF=False, device="cpu", compile=False).to(config["device"]).eval()
        self.processor = Sam3Processor(self.model, device=config["device"],
                                       confidence_threshold=config.get("confidence_threshold", 0.25))

    def propose(self, rgb, queries):
        import torch
        from PIL import Image
        proposals = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(rgb))
            for query in queries:
                self.processor.reset_all_prompts(state)
                answer = self.processor.set_text_prompt(query, state)
                masks = answer["masks"].squeeze(1).cpu().numpy().astype(bool)
                scores = answer["scores"].float().cpu().numpy()
                for mask, score in zip(masks, scores):
                    box = mask_box(mask)
                    if box:
                        proposals.append({"bbox": box, "score": float(score), "query": query, "mask": mask})
        return sorted(proposals, key=lambda item: -item["score"])


class Sam:
    def __init__(self, config):
        self.config = config
        self.model = self.predictor = None
        self.metadata = {}

    def _load(self):
        if not self.config.get("checkpoint_path"):
            raise ValueError("Configure a local SAM3 checkpoint_path before model inference")
        import hashlib
        import inspect
        import torch
        from sam3.model_builder import build_sam3_video_model
        torch.manual_seed(self.config.get("seed", 42))
        path = Path(self.config["checkpoint_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("Set checkpoint_path to local SAM3 (not SAM3.1) weights")
        self.model = build_sam3_video_model(checkpoint_path=str(path), load_from_HF=False,
                                          device=self.config["device"], compile=False)
        self.predictor = self.model.tracker
        self.predictor.backbone = self.model.detector.backbone
        with path.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        implementation = Path(inspect.getfile(type(self.predictor)))
        self.metadata = {"checkpoint_path": str(path), "checkpoint_sha256": checkpoint_hash,
                         "tracker_code_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
                         "seed": self.config.get("seed", 42),
                         "mode": "sam3_explicit_instance_tracking"}

    def track(self, frame_directory, *, seed_frame, bbox, width, height, count):
        if self.predictor is None:
            self._load()
        import torch
        device = torch.device(self.config["device"])
        relative_box = np.asarray(bbox, dtype=np.float32).reshape(1, 4) / [width, height, width, height]
        # SAM3's official SAM2-task example uses normalized boxes and the instance tracker.
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # CPU state offload can read unfinished asynchronous D2H copies in SAM3's consolidation.
            state = self.predictor.init_state(video_path=str(frame_directory), offload_video_to_cpu=True,
                                             offload_state_to_cpu=False)
            if state["num_frames"] != count:
                raise ValueError("SAM input frame count differs from the source mapping")
            self.predictor.add_new_points_or_box(inference_state=state, frame_idx=seed_frame, obj_id=1,
                                                box=relative_box, rel_coordinates=True)
            for reverse in (False, True):
                if reverse and seed_frame == 0:
                    continue
                for frame, ids, _, masks, _ in self.predictor.propagate_in_video(
                    state, start_frame_idx=seed_frame, max_frame_num_to_track=count,
                    reverse=reverse, propagate_preflight=True,
                ):
                    if reverse and frame == seed_frame:
                        continue
                    # Returned IDs, not initial prompt ordering, select the mask.
                    slot = list(ids).index(1)
                    mask = (masks[slot] > 0).squeeze(0).cpu().numpy().astype(bool)
                    if mask.shape != (height, width):
                        raise ValueError("SAM returned a mask at a different spatial resolution")
                    yield int(frame), mask
