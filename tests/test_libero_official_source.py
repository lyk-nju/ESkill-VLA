"""Official action correspondence must retain grasp changes and refuse approximate matches."""
import importlib
import numpy as np

source=importlib.import_module('scripts.preprocess.libero.libero_40.source')
align=importlib.import_module('scripts.preprocess.libero.libero_40.align')


def test_noop_filter_keeps_stationary_gripper_changes_and_original_indices():
    actions=np.asarray([[0,0,0,0,0,0,-1], [1,0,0,0,0,0,-1],
                        [0,0,0,0,0,0,-1], [0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1], [2,0,0,0,0,0,1]],dtype=np.float32)
    indices=source.retained_indices(actions)
    np.testing.assert_array_equal(indices,[1,3,5])
    mapped,method=source.match_indices(actions,actions[indices])
    np.testing.assert_array_equal(mapped,[1,3,5])
    assert method=='noop_filtered_exact'


def test_exact_raw_sequence_does_not_drop_rows_or_shift_source_indices():
    actions=np.asarray([[0,0,0,0,0,0,1],[1,0,0,0,0,0,-1]],dtype=np.float32)
    mapped,method=source.match_indices(actions,actions.copy())
    np.testing.assert_array_equal(mapped,[0,1])
    assert method=='raw_exact'


def test_nearly_equal_actions_and_subsequence_are_not_accepted():
    actions=np.zeros((4,7),np.float32); actions[:,0]=[1,2,3,4]
    perturbed=actions.copy(); perturbed[2,0]+=1e-5
    assert source.match_indices(actions,perturbed)==(None,None)
    assert source.match_indices(actions,actions[1:])==(None,None)


def test_offset_diagnostic_does_not_silently_retime_trajectory():
    data={'joints':np.tile(np.arange(5)[:,None],(1,7)),
          'state':np.c_[np.arange(5),np.zeros((5,7))]}
    track={'joints':data['joints']+1,'eef':data['state'][:,:3]+[1,0,0]}
    result=align.robot_metrics(track,data)
    assert result['offset_checks']['-1']['eef_max_m']==0
    assert result['same_row']['eef_max_m']==1
    assert result['offset_applied']==0
