"""Two stored RGB frames with Pick/Place, distractors and native visible masks."""
import json

import numpy as np

from openpi.augmentation import GroundingAugmentation
from openpi.policies.grounding_dataset import GroundingDataset
from openpi.training.config import TrainConfig
from openpi.models.pi0_config import ESkillConfig

class ToyTokenizer:
    tokenizer = property(lambda self: self)

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def eos_id(self):
        return 2

    def decode(self, tokens):
        return ''.join(chr(t) for t in tokens)

    def __call__(self, text, state=None):
        tokens = np.array(self.encode(text)[:96], np.int64)
        return np.pad(tokens, (0, 96-len(tokens))), np.arange(96) < len(tokens)


def write_scene(tmp_path):
    def ref(key, **kw):
        return dict(uri='arrays.npz', key=key, **kw)
    masks = np.zeros((2,32,32),bool)
    masks[:,2:10,2:10] = True
    arrays = dict(frames=np.arange(2), valid=np.ones(2,bool), rgb=np.full((2,32,32,3),180,np.uint8),
        milk=np.tile([2,2,10,10],(2,1)), basket=np.tile([20,20,28,28],(2,1)), pudding=np.tile([24,2,30,8],(2,1)),
        gripper=np.tile([12,12,16,16],(2,1)),
        visible=np.packbits(masks.reshape(2,-1),axis=1), visible_fraction=np.ones(2,np.float32),
        partially_visible=np.full(2,.85,np.float32),
        contact=np.full((2,4),np.nan), eligible=np.zeros(2,bool))
    np.savez(tmp_path/'arrays.npz',**arrays)
    entities = {key:dict(kind='object',category=key,part='whole',visible_fraction=ref('visible_fraction'),regions={'image':dict(frame_indices=ref('frames'),
        boxes_xyxy=ref(key),boxes_valid=ref('valid'),visible_masks=ref('visible',encoding='packbits',shape=[2,32,32]))})
        for key in ('milk','basket','pudding')}
    entities['pudding']['category'] = 'chocolate_pudding'
    entities['pudding']['copy_eligible'] = ref('valid')
    for key in ('milk', 'basket'):
        entities[key]['visible_fraction'] = ref('partially_visible')
    for entity in entities.values():
        entity['interaction_locked'] = ref('eligible')  # All False; separate from copy eligibility.
    meta = dict(episode_id='test',instructions=['put the milk in the basket'],scene_entities=entities,
        cameras={'image':dict(frame_indices=ref('frames'),valid=ref('valid'),rgb=ref('rgb'))},
        online_grounding=dict(interaction_boxes=ref('contact')),
        effector_regions={'single':{'image':dict(frame_indices=ref('frames'),boxes_xyxy=ref('gripper'),boxes_valid=ref('valid'))}},
        events=[dict(arm='single',skill='Pick' if i==0 else 'Place',start_frame=i,stop_frame=i+1,interaction_target=dict(entity_id=key),
                     grounding_template='pick up the {object}' if i==0 else None,
                     description='pick up the milk' if i==0 else 'place the milk in the basket')
                for i,key in enumerate(['milk','basket'])])
    (tmp_path/'episode.json').write_text(json.dumps(meta))
    (tmp_path/'index.json').write_text(json.dumps(dict(episodes=[dict(episode_id='test',path='episode.json')])))


def visual_dataset(tmp_path, *, goal=False, **aug):
    write_scene(tmp_path)
    path = tmp_path / 'episode.json'
    meta = json.loads(path.read_text())
    with np.load(tmp_path / 'arrays.npz') as archive:
        arrays = dict(archive)
    arrays['rgb'] = np.tile(np.arange(32, dtype=np.uint8)[None, None, :, None] * 6, (2, 32, 1, 3))
    for entity, annotation in meta['scene_entities'].items():
        mask = np.zeros((2, 32, 32), np.uint8)
        x, y, r, b = arrays[entity][0]
        mask[:, y:b, x:r] = 1
        key = entity + '_visible'
        arrays[key] = np.packbits(mask.reshape(2, -1), axis=1)
        annotation['regions']['image']['visible_masks']['key'] = key
    np.savez(tmp_path / 'arrays.npz', **arrays)
    if goal:
        # A second training episode witnesses bowl -> plate with the same placement relation.
        meta['scene_entities']['milk']['category'] = 'akita_black_bowl'
        meta['scene_entities']['basket']['category'] = 'flat_stove'
        meta['scene_entities']['pudding']['category'] = 'plate'
        meta['instructions'] = ['put the bowl on the stove']
        second = json.loads(json.dumps(meta))
        second['episode_id'] = 'second'
        second['instructions'] = ['put the bowl on the plate']
        second['events'][1]['interaction_target']['entity_id'] = 'pudding'
        (tmp_path / 'second.json').write_text(json.dumps(second))
        (tmp_path / 'index.json').write_text(json.dumps(dict(episodes=[
            dict(episode_id='test', path='episode.json'),dict(episode_id='second', path='second.json')])))
    path.write_text(json.dumps(meta))
    model = ESkillConfig.tiny(vocab_size=128, current_grounding='visual_primitive', attention_layers=(0,))
    cfg = TrainConfig(str(tmp_path/'index.json'), '', '', '', model=model, training_stage='visual_primitive',
                      grounding_augmentation=GroundingAugmentation(**aug), loss_weights={'current_align': .1})
    return GroundingDataset(cfg.data_index, cfg, ToyTokenizer())
