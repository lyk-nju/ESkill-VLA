"""Explicit assistant annotations must preserve the automatic record and episode identity."""
from copy import deepcopy
import importlib

import pytest

manual=importlib.import_module('scripts.preprocess.libero.atomic_vla.manual_binding12')


def fixture():
    sample={'annotation_id':7,'key':'episode-a','source_cache_sha256':'rgb-state-a'}
    rows=[{'source_id':'known','status':'rule_bound','instance':'pot_b','scope':{'part':'whole'},
           'start_frame':0,'stop_frame':5,'goal_frame':4,'skill':'Pick','evidence':[]},
          {'source_id':'missing','status':'ambiguous','reason':'multiple_targets','instance':None,'scope':None,
           'start_frame':5,'stop_frame':10,'goal_frame':9,'skill':'Place','evidence':[]}]
    patch={'annotation_id':7,'key':'episode-a','source_cache_sha256':'rgb-state-a','event':'missing',
           'start_frame':5,'stop_frame':10,'goal_frame':9,'skill':'Place','instance':'pot_b','scope':{'part':'whole'},
           'geometry':{'status':'resolved'},'evidence_frames':[5,8,9],'observation':'same held pot in source frames'}
    return sample,rows,{'events':[patch]}


def test_manual_fill_keeps_automatic_input_and_known_binding():
    sample,rows,record=fixture();before=deepcopy(rows)
    result=manual.apply_manual(sample,rows,record)
    assert rows==before and result[0]==before[0]
    assert result[1]['status']=='manual_bound' and result[1]['instance']=='pot_b'
    assert result[1]['automatic_result']['status']=='ambiguous'
    assert result[1]['manual_assignment']['user_review']=='pending'
    assert result[1]['goal_frame']==9


def test_manual_fill_rejects_a_different_source_cache():
    sample,rows,record=fixture();sample['source_cache_sha256']='different-episode-content'
    with pytest.raises(ValueError,match='different source episode'):manual.apply_manual(sample,rows,record)


def test_manual_fill_cannot_overwrite_automatic_success_or_new_boundaries():
    sample,rows,record=fixture();rows[1]['status']='rule_bound'
    with pytest.raises(ValueError,match='only fill unresolved'):manual.apply_manual(sample,rows,record)
    rows[1]['status']='ambiguous';rows[1]['goal_frame']=8
    with pytest.raises(ValueError,match='bounds or skill changed'):manual.apply_manual(sample,rows,record)
