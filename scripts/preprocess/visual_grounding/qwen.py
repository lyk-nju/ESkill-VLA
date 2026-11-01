"""One local Qwen3-VL client, shared by text annotation and visual grounding."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_json(text):
    value = text.strip()
    if value.startswith("```json\n") and value.endswith("```"):
        value = value[8:-3].strip()
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("Qwen must return one JSON object")
    return result


def pixel_box(box, width, height):
    """Qwen's explicit [0,1000] xyxy contract -> original-image pixel xyxy."""
    values = np.asarray(box, dtype=float)
    if (values.shape != (4,) or not np.isfinite(values).all()
            or not (0 <= values[0] < values[2] <= 1000 and 0 <= values[1] < values[3] <= 1000)):
        raise ValueError("Expected finite ordered bbox_1000, not guessed pixel coordinates")
    return (values * [width/1000, height/1000, width/1000, height/1000]).tolist()


class Qwen:
    def __init__(self, config):
        self.config = config
        self.model = self.processor = None
        self.metadata = {}
        self.last_raw_response = None

    def _load(self):
        if not self.config.get("model_path"):
            raise ValueError("Configure a local Qwen3-VL model_path before model inference")
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch.manual_seed(self.config.get("seed", 42))
        path = Path(self.config["model_path"]).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError("Set model_path to a local Qwen3-VL dense Instruct checkpoint")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to(self.config["device"]).eval()
        self.metadata = {"model_path": str(path), "transformers": transformers.__version__,
                         "torch": torch.__version__, "seed": self.config.get("seed", 42),
                         "model_config": self.model.config.to_dict()}

    def ask(self, prompt, images):
        """images is [(source frame/camera label, uint8 RGB), ...]. No remote calls."""
        if self.model is None:
            self._load()
        import torch
        content = [{"type": "text", "text": prompt}]
        for label, rgb in images:
            content.extend([{"type": "text", "text": label},
                            {"type": "image", "image": Image.fromarray(rgb)}])
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False,
                                            max_new_tokens=self.config["max_new_tokens"])
        tokens = generated[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
        self.last_raw_response = raw
        return parse_json(raw), raw

    def locate(self, description, images):
        prompt = (
            "Locate this specific physical instance, not every object of its category: " + description +
            '\nChoose ONE of the supplied labeled frames where it is visibly identifiable. '
            'Return JSON only: {"visible":true,"frame_index":12,"bbox_1000":[x1,y1,x2,y2],'
            '"reason":"brief evidence"}. Coordinates are xyxy normalized to 0..1000. '
            'For a gripper include its palm/fingers, not the full robot arm. Robot left/right means '
            'physical arm identity, NOT image left/right. If identity or visibility is uncertain, '
            'return {"visible":false,"frame_index":null,"bbox_1000":null,"reason":"why"}. '
            'Do not hallucinate an occluded object or substitute its destination/container.'
        )
        answer, raw = self.ask(prompt, [(f"frame_index={frame}", rgb) for frame, rgb in images])
        if type(answer.get("visible")) is not bool:
            raise ValueError("Localization requires an explicit visible boolean")
        if answer["visible"]:
            candidates = dict(images)
            frame = answer["frame_index"]
            if type(frame) is not int or frame not in candidates:
                raise ValueError("Qwen selected a frame that was not supplied")
            height, width = candidates[frame].shape[:2]
            answer["bbox"] = pixel_box(answer["bbox_1000"], width, height)
        return answer, raw
