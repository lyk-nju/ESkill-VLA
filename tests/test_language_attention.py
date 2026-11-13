"""Word spans refer to actual model tokens; masks and full softmax denominator are retained."""
import numpy as np
import pytest
import sentencepiece
import torch

from scripts.analysis.language_attention import LanguageCapture, token_layout
from test_attention_viewer import checkpoints


def test_phrase_alignment_repetition_missing_and_truncation(checkpoints):
    root, _ = checkpoints
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(root / 'tokenizer.model'))
    text = 'pick cup then place cup'
    ids = tokenizer.encode(text, add_bos=True)
    layout = token_layout(tokenizer, text, ids, [True] * len(ids), object_word='cup', target_word='cup')
    assert layout['object']['span'] == [5, 8]
    assert layout['target']['span'] == [20, 23]
    assert set(layout['object']['indices']).isdisjoint(layout['target']['indices'])
    assert 0 not in layout['instruction']['indices']
    for name in ('object', 'target'):
        assert tokenizer.decode(layout[name]['token_ids']).strip() == 'cup'
    truncated = token_layout(tokenizer, text, ids[:-2], [True] * (len(ids)-2),
                             object_word='cu', target_word='cup')
    assert truncated['object']['status'] == 'PHRASE NOT IN INSTRUCTION'
    assert truncated['target']['status'] == 'PHRASE TRUNCATED'
    assert truncated['instruction']['status'] == 'PHRASE TRUNCATED'
    assert not truncated['target']['indices']
    with pytest.raises(ValueError, match='actual VLM input'):
        token_layout(tokenizer, text, [0] * len(ids), [True] * len(ids))


def test_word_queries_use_exact_softmax_with_masked_keys(checkpoints):
    root, _ = checkpoints
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(root / 'tokenizer.model'))
    text = 'pick cup then place it'
    ids = tokenizer.encode(text, add_bos=True) + [0, 0]
    valid = torch.ones(1, len(ids), dtype=torch.bool)
    valid[0, -2:] = False
    observer = LanguageCapture(tokenizer, text, object_word='cup', target_word='it')
    observer.prepare(dict(images=torch.zeros(1, 2, 3, 32, 32), tokens=torch.tensor([ids]), token_mask=valid), 32)
    generator = torch.Generator().manual_seed(12)
    n = len(ids) + 32
    q = torch.randn(1, 4, n, 8, generator=generator)
    k = torch.randn(1, 1, n, 8, generator=generator)
    mask = torch.cat((torch.ones(1, 32, dtype=torch.bool), valid), -1)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        observer.record(q, k, mask)
    result = observer.result()
    assert result['valid'].all()
    assert tokenizer.decode(result['layout']['target']['token_ids']).strip() == 'it'
    scores = q @ k.repeat_interleave(4, 1).transpose(-1, -2) / 8 ** .5
    full = scores.masked_fill(~mask[:, None, None], float('-inf')).softmax(-1)
    for i, name in enumerate(('target', 'object', 'instruction')):
        rows = [j + 32 for j in observer.layout[name]['indices']]
        expected = full[0, :, rows].mean((0, 1)).numpy()
        np.testing.assert_allclose(result['probabilities'][i], expected, atol=1e-7)
    assert not result['probabilities'][:, -2:].any()
    np.testing.assert_allclose(result['maps'].sum((-1, -2)), 1., atol=1e-6)
    np.testing.assert_allclose(result['view_mass'].sum(1), result['probabilities'][:, :32].sum(1), atol=1e-6)
