"""Read a real unified-data checkpoint through the LIBERO evaluation adapter."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from openpi.policies.eskill_dataset import EpisodeReader
from openpi.policies.libero_policy import LiberoWirePolicy
from openpi.policies.policy_config import create_trained_policy


def main(path):
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = saved['config']
    step = saved['step']
    del saved
    policy = create_trained_policy(path, device='cuda:0')
    wire = LiberoWirePolicy(policy, config, steps=10)
    index = Path(config['data_index'])
    entries = json.loads(index.read_text())['episodes']
    records = []
    for number in (0, len(entries)//2, len(entries)-1):
        reader = EpisodeReader(index.parent / entries[number]['path'])
        frame = reader.meta['frame_count']//2
        seq = reader.meta['state']
        state = reader.array(seq['values'])[reader.row(seq,frame)]
        observation = {'observation/state':state[-8:], 'observation/joint_state':state[:7],
                       'prompt':reader.meta['instructions'][0]}
        observation.update({'observation/'+c:reader.rgb(c,frame) for c in wire.cameras})
        result = wire.infer(observation)
        assert result['actions'].shape == (config['model']['action_horizon'],7)
        assert np.isfinite(result['actions']).all()
        assert (result['future_skills'][1] == -1).all()
        records.append(dict(episode_id=reader.meta['episode_id'],frame=frame,
                            action_shape=list(result['actions'].shape), finite=True))
    report = dict(checkpoint=str(path),step=step,cameras=wire.cameras,
                  requires_joint_state=wire.requires_joint_state,
                  training_vae_absent=policy.model.visual_supervision is None,
                  samples=records, closed_loop_evaluation=False)
    (path.parent.parent/'policy_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
