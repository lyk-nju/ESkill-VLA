"""Instruction-word queries over image keys, observed in the unchanged VLM prefill."""
import math
import re

import numpy as np
import torch


GROUPS = ('target', 'object', 'instruction')


def token_layout(tokenizer, instruction, ids, valid, *, object_word=None, target_word=None):
    """Align character spans to actual SentencePiece positions, excluding BOS/padding/newline.

    Object uses the first exact phrase occurrence, Target the last. Missing or truncated
    phrases stay unavailable; no object identity or destination is guessed from labels.
    """
    text = instruction.strip().replace('_', ' ').replace('\n', ' ')
    # Keep the proto alive: its piece objects reference storage owned by the proto.
    encoded = tokenizer.encode(text, out_type='immutable_proto')
    pieces = [(piece.begin, piece.end) for piece in encoded.pieces]
    expected = tokenizer.encode(text, add_bos=True)
    count = min(len(expected), len(ids))
    if list(ids[:count]) != expected[:count]:
        raise ValueError('Instruction tokens do not match the actual VLM input')
    result = {}
    for name, phrase in (('target', target_word), ('object', object_word), ('instruction', text)):
        item = dict(phrase=phrase, indices=[], status='PHRASE NOT SPECIFIED')
        if phrase:
            phrase = phrase.strip().replace('_', ' ').replace('\n', ' ')
            matches = list(re.finditer(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text, re.IGNORECASE))
            if not matches:
                item['status'] = 'PHRASE NOT IN INSTRUCTION'
            else:
                match = matches[-1] if name == 'target' else matches[0]
                positions = [i + 1 for i, (begin, end) in enumerate(pieces)
                             if begin < match.end() and end > match.start()]
                item.update(phrase=text[match.start():match.end()], span=list(match.span()))
                if not positions or any(i >= len(ids) or not valid[i] for i in positions):
                    item['status'] = 'PHRASE TRUNCATED'
                else:
                    item.update(indices=positions, token_ids=[int(ids[i]) for i in positions],
                                pieces=[tokenizer.id_to_piece(int(ids[i])) for i in positions], status='OK')
        result[name] = item
    return result


class LanguageCapture:
    """Reconstruct selected word-query probabilities without replacing the native attention."""
    def __init__(self, tokenizer, instruction, *, object_word=None, target_word=None):
        self.tokenizer, self.instruction = tokenizer, instruction
        self.words = dict(object_word=object_word, target_word=target_word)

    def prepare(self, inputs, image_keys):
        self.image_keys = image_keys
        self.views = inputs['images'].shape[1]
        self.layout = token_layout(self.tokenizer, self.instruction, inputs['tokens'][0].tolist(),
                                   inputs['token_mask'][0].tolist(), **self.words)
        self.records = []

    def record(self, q, k, mask):
        # Only selected rows are reconstructed, preserving the full softmax key denominator.
        rows = sorted({i + self.image_keys for group in self.layout.values() for i in group['indices']})
        if not rows:
            return
        key = k[0].repeat_interleave(q.shape[1] // k.shape[1], 0).float()
        scores = q[0, :, rows].float() @ key.transpose(-1, -2) / math.sqrt(q.shape[-1])
        probabilities = scores.masked_fill(~mask[0, None, None], float('-inf')).softmax(-1).mean(0)
        self.records.append(torch.stack([
            probabilities[[rows.index(i + self.image_keys) for i in self.layout[name]['indices']]].mean(0)
            if self.layout[name]['indices'] else probabilities.new_zeros(probabilities.shape[-1])
            for name in GROUPS]).cpu().numpy())

    def result(self):
        if not self.records:
            raise RuntimeError('No language attention captured')
        probabilities = np.stack(self.records).mean(0)
        grid = math.isqrt(self.image_keys // self.views)
        raw = probabilities[:, :self.image_keys].reshape(3, self.views, grid, grid)
        mass = raw.sum((-1, -2))
        maps = raw / np.maximum(mass[..., None, None], 1e-12)
        return dict(maps=maps, view_mass=mass, probabilities=probabilities,
                    valid=np.array([bool(self.layout[name]['indices']) for name in GROUPS]), layout=self.layout)
