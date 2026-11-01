"""Match a scene-state excerpt to case01, then render visible native object masks."""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.preprocess.libero.rlds import load_records
from scripts.preprocess.libero.render import Region, overlay, GREEN, BLUE
from scripts.preprocess.artifacts import digest
from .native_gripper import bbox

from scripts.preprocess import DATA_ROOT

ROOT = DATA_ROOT / "libero/atomic-vla"
OUT = ROOT/'check/test/native_object'
FRAMES = [0, 48, 96]
MIRROR = 'clip-rt/modified_libero_hdf5'
REVISION = '6a6659f8ac7d580fd594173a0e3abf880c843130'
FILENAME = 'libero_spatial_no_noops/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5'
EXCERPT = Path('/data1/yuankai/robot-research/data/external/atomic_vla/native_object')/REVISION/'case01_scene_excerpt.npz'


def source():
    path = ROOT/'check/raw/manifest.json'
    manifest = json.loads(path.read_text())
    sample = next(s for s in manifest['samples'] if s['number'] == 1)
    assert sample['key'] == 'libero_spatial_no_noops_s007_r022' and sample['frames'] == 97
    raw = Path(manifest['raw_root'])
    assert digest(raw/sample['shard']) == next(f['sha256'] for f in manifest['files'] if f['path'] == sample['shard'])
    _, data = next(load_records(raw, [sample]))
    return path, sample, data


class RangeFile(io.RawIOBase):
    """Bounded, cached HTTP reads; never allow a fallback whole-file download."""
    def __init__(self, url, size, budget=256*1024**2, block_size=256*1024):
        import requests
        self.session = requests.Session()
        self.url, self.size, self.budget, self.block_size = url, size, budget, block_size
        self.position, self.transferred, self.requests = 0, 0, 0
        self.blocks = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if self.position < 0: raise ValueError('negative seek')
        return self.position

    def block(self, index):
        if index not in self.blocks:
            start = index*self.block_size
            stop = min(start+self.block_size, self.size)-1
            size = stop-start+1
            if self.transferred+size > self.budget: raise RuntimeError('Range download budget exceeded')
            response = self.session.get(self.url+f'?download=true&read_block={index}',
                headers={'Range':f'bytes={start}-{stop}'}, timeout=(20,60), stream=True)
            with response:
                response.raise_for_status()
                assert response.status_code == 206, 'Server did not honor Range; refusing full file'
                assert response.headers['Content-Range'] == f'bytes {start}-{stop}/{self.size}'
                content = response.raw.read(size+1)
                assert len(content) == size
            self.transferred += size; self.requests += 1
            self.blocks[index] = content
        return self.blocks[index]

    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data
        return len(data)

    def read(self, count=-1):
        count = self.size-self.position if count < 0 else min(count,self.size-self.position)
        pieces = []
        while count > 0:
            index, offset = divmod(self.position, self.block_size)
            piece = self.block(index)[offset:offset+count]
            pieces.append(piece); self.position += len(piece); count -= len(piece)
        return b''.join(pieces)


def fetch():
    import h5py
    path, sample, data = source()
    actions = np.asarray(data['steps/action'],np.float32).reshape(97,7)
    joints = np.asarray(data['steps/observation/joint_state'],np.float32).reshape(97,7)
    state = np.asarray(data['steps/observation/state'],np.float32).reshape(97,8)
    url = f'https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/{FILENAME}'
    remote = RangeFile(url, 1774379312)
    scanned, matches = [], []
    with h5py.File(remote, 'r') as h:
        for name, group in h['data'].items():
            n = group['actions'].shape[0]
            match = n == 97 and np.array_equal(group['actions'][()].astype(np.float32),actions)
            scanned.append({'demo':name,'frames':n,'all_actions_equal_float32':match})
            if match: matches.append(name)
            if len(scanned)%10==0:print(json.dumps({'scanned':len(scanned),'matches':matches,'download_bytes':remote.transferred}),flush=True)
        assert len(matches)==1, matches
        demo=matches[0];group=h['data'][demo]
        states=group['states'][()]
        mirror_joints=group['obs/joint_states'][()].astype(np.float32)
        mirror_state=np.concatenate([group['obs/ee_states'][()],group['obs/gripper_states'][()]],axis=1).astype(np.float32)
        rgb=np.stack([group['obs/agentview_rgb'][frame] for frame in FRAMES])
        comparisons=[]
        for index,frame in enumerate(FRAMES):
            original=np.asarray(Image.open(io.BytesIO(data['steps/observation/image'][frame])).convert('RGB'))
            rotated=rgb[index][::-1,::-1]
            comparisons.append({'frame':frame,'rgb_mae_after_180_rotation':float(np.abs(rotated.astype(float)-original).mean())})
        EXCERPT.parent.mkdir(parents=True,exist_ok=True)
        with EXCERPT.open('xb') as stream:
            np.savez_compressed(stream,states=states,actions=actions,joints=joints,state=state,
                mirror_joints=mirror_joints,mirror_state=mirror_state,mirror_rgb=rgb,frames=FRAMES)
        attrs={k:str(v) for k,v in group.attrs.items()}
    report={'source_manifest_sha256':digest(path),'source_key':sample['key'],'mirror':MIRROR,'revision':REVISION,
        'remote_file':FILENAME,'remote_file_size':remote.size,
        'remote_lfs_sha256_from_api':'d2425433c67d1aebbe9a07a5f4aa3fb4674c28e270e406e1debe80f0b3fd897b',
        'full_remote_file_hash_verified':False,'range_bytes':remote.transferred,'range_requests':remote.requests,
        'scanned':scanned,'matched_demo':demo,'actions_exact_equal_float32':True,
        'max_joint_difference':float(np.abs(mirror_joints-joints).max()),
        'max_proprio_difference':float(np.abs(mirror_state-state).max()),
        'rgb_comparisons':comparisons,'states_shape':list(states.shape),'group_attrs':attrs,
        'excerpt_path':str(EXCERPT),'excerpt_sha256':digest(EXCERPT),'manual_review':'pending'}
    (OUT/'source_match.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['scanned','group_attrs']}),flush=True)


def render(libero_root):
    sys.path.insert(0,str(libero_root))
    from libero.libero.envs import OffScreenRenderEnv
    import mujoco
    import robosuite.macros as macros
    matched=json.loads((OUT/'source_match.json').read_text())
    assert digest(EXCERPT)==matched['excerpt_sha256']
    path,sample,data=source()
    assert digest(path)==matched['source_manifest_sha256']
    excerpt=np.load(EXCERPT)
    bddl=libero_root/'libero/libero/bddl_files/libero_spatial'/Path(FILENAME).name.replace('_demo.hdf5','.bddl')
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_names=['agentview'],camera_heights=256,camera_widths=256,render_gpu_device_id=0)
    env.seed(0);env.reset()
    sim=env.sim
    # BDDL names this bowl as the object on cookies; do not substitute the other bowl.
    identity='akita_black_bowl_1'
    obj=env.env.objects_dict[identity]
    ids=[sim.model.geom_name2id(n) for n in obj.visual_geoms]
    gripper_ids=[sim.model.geom_name2id(n) for n in env.robots[0].gripper.visual_geoms]
    sim._render_context_offscreen.vopt.sitegroup[:]=0
    results=[];panels=[]
    def to_rlds(value):
        convention=1 if macros.IMAGE_CONVENTION=='opengl' else -1
        return value[::convention][::-1,::-1].copy()
    for index,frame in enumerate(FRAMES):
        sim.set_state_from_flattened(excerpt['states'][frame]);sim.forward()
        rebuilt=to_rlds(sim.render(width=256,height=256,camera_name='agentview'))
        seg=to_rlds(sim.render(width=256,height=256,camera_name='agentview',segmentation=True))
        mask=(seg[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[...,1],ids)
        gripper=(seg[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[...,1],gripper_ids)
        box=bbox(mask)
        original=np.asarray(Image.open(io.BytesIO(data['steps/observation/image'][frame])).convert('RGB'))
        Image.fromarray(rebuilt).save(OUT/f'frame_{frame:03d}_scene_rgb.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(OUT/f'frame_{frame:03d}_object_mask.png')
        preview=overlay(original,Region(box,mask),GREEN)
        preview.save(OUT/f'frame_{frame:03d}_object_overlay.png')
        rgb_error=float(np.abs(rebuilt.astype(float)-original).mean())
        mirror_error=float(np.abs(rebuilt.astype(float)-excerpt['mirror_rgb'][index][::-1,::-1]).mean())
        results.append({'frame':frame,'object':identity,'bbox':box,'mask_pixels':int(mask.sum()),
            'scene_rgb_mae_vs_source':rgb_error,'scene_rgb_mae_vs_mirror':mirror_error,
            'object_world_position':sim.data.get_body_xpos(obj.root_body).tolist(),
            'visible_gripper_bbox_for_alignment':bbox(gripper),'manual_review':'pending'})
        panels.append([Image.fromarray(original),Image.fromarray(rebuilt),preview])
        print(json.dumps(results[-1]),flush=True)
    env.close()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    board=Image.new('RGB',(1152,1392),'#171b22');draw=ImageDraw.Draw(board)
    draw.text((12,10),'Case01 | native visible object mask | green = task bowl | review pending',font=font,fill='white')
    for c,title in enumerate(['SOURCE MAIN RGB','RESTORED SCENE RGB','OBJECT MASK + BBOX']):draw.text((c*384+12,46),title,font=font,fill='white')
    for i,(frame,images) in enumerate(zip(FRAMES,panels)):
        y=82+i*434
        draw.text((12,y),f'Frame {frame} | {identity} | bbox {results[i]["bbox"]}',font=font,fill=GREEN)
        for c,img in enumerate(images):board.paste(img.resize((384,384)),(c*384,y+30))
    board.save(OUT/'overview.png')
    report={'source_match_sha256':digest(OUT/'source_match.json'),'script_sha256':digest(Path(__file__)),
        'source_key':sample['key'],'object_instance':identity,'visual_geom_names':obj.visual_geoms,'visual_geom_ids':ids,
        'full_scene_state_used':True,'external_occlusion_preserved':True,'manual_calibration_applied':False,
        'image_transform':'robosuite observable convention then OpenVLA 180-degree rotation',
        'results':results,'manual_review':'pending'}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


def render_full(libero_root, out=None):
    """Render each target independently, keeping the source pose and camera fixed."""
    sys.path.insert(0,str(libero_root))
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from .bind_instances import RULES_PATH, bind_events, resolve_geometry
    from .source import source_intervals
    import mujoco
    import robosuite.macros as macros
    out=out or OUT.parent/'native_object_full'
    out.mkdir(parents=True,exist_ok=True)
    matched=json.loads((OUT/'source_match.json').read_text())
    assert digest(EXCERPT)==matched['excerpt_sha256']
    path,sample,data=source()
    assert digest(path)==matched['source_manifest_sha256']
    excerpt=np.load(EXCERPT)
    bddl=libero_root/'libero/libero/bddl_files/libero_spatial'/Path(FILENAME).name.replace('_demo.hdf5','.bddl')
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_names=['agentview'],camera_heights=256,camera_widths=256,render_gpu_device_id=0)
    env.seed(0);env.reset()
    sim=env.sim
    rules=json.loads(RULES_PATH.read_text())
    bindings=bind_events(robosuite_parse_problem(str(bddl)),
        source_intervals(sample['annotation'],sample['frames']),sample['task'],rules)
    for binding in bindings:
        binding['geometry']=resolve_geometry(env,binding)
    context=sim._render_context_offscreen
    original_groups=sim.model.geom_group.copy()
    original_options=context.vopt.geomgroup.copy()
    context.vopt.sitegroup[:]=0
    results=[];panels=[]

    def segmentation():
        convention=1 if macros.IMAGE_CONVENTION=='opengl' else -1
        value=sim.render(width=256,height=256,camera_name='agentview',segmentation=True)
        return value[::convention][::-1,::-1].copy()

    def target_mask(seg,selected):
        return (seg[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[...,1],selected)

    for frame in FRAMES:
        binding=next(e for e in bindings if e['start_frame']<=frame<e['stop_frame'])
        identity=binding['instance']
        names={'gripper':list(env.robots[0].gripper.visual_geoms),
               'object':binding['geometry']['visual_geom_names']}
        ids={role:[sim.model.geom_name2id(n) for n in values] for role,values in names.items()}
        sim.set_state_from_flattened(excerpt['states'][frame]);sim.forward()
        scene=segmentation()
        visible={role:target_mask(scene,selected) for role,selected in ids.items()}
        complete={}
        for role,selected in ids.items():
            sim.model.geom_group[:]=5;sim.model.geom_group[selected]=1
            context.vopt.geomgroup[:]=0;context.vopt.geomgroup[1]=1
            seg=segmentation()
            rendered_ids=np.unique(seg[...,1][seg[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM)]).tolist()
            assert set(rendered_ids).issubset(selected)
            complete[role]=target_mask(seg,selected)
            sim.model.geom_group[:]=original_groups
            context.vopt.geomgroup[:]=original_options
        original=np.asarray(Image.open(io.BytesIO(data['steps/observation/image'][frame])).convert('RGB'))
        row={'frame':frame,'event_id':binding['source_id'],'object_instance':identity,
             'object_part':binding['scope']['part'],'visual_geom_names':names,'regions':{}}
        full_preview=Image.fromarray(original)
        for role,color in [('gripper',BLUE),('object',GREEN)]:
            mask=complete[role];reference=visible[role]
            outside=int((reference&~mask).sum())
            mask_name=f'frame_{frame:03d}_{role}_full_mask.png'
            visible_name=f'frame_{frame:03d}_{role}_visible_mask.png'
            Image.fromarray(mask.astype(np.uint8)*255).save(out/mask_name)
            Image.fromarray(reference.astype(np.uint8)*255).save(out/visible_name)
            row['regions'][role]={'bbox':bbox(mask),'mask':mask_name,'region_extent':'full_in_frame',
                'full_pixels':int(mask.sum()),'reference_visible_bbox':bbox(reference),
                'reference_visible_mask':visible_name,'reference_visible_pixels':int(reference.sum()),
                'external_occlusion_pixels_recovered':int((mask&~reference).sum()),
                'visible_pixels_outside_full':outside,'rendered_geom_ids':ids[role]}
            full_preview=overlay(np.asarray(full_preview),Region(bbox(mask),mask),color)
        full_preview.save(out/f'frame_{frame:03d}_full_overlay.png')
        old=overlay(original,Region(bbox(visible['object']),visible['object']),GREEN)
        old.save(out/f'frame_{frame:03d}_visible_object_overlay.png')
        new=overlay(original,Region(bbox(complete['object']),complete['object']),GREEN)
        new.save(out/f'frame_{frame:03d}_full_object_overlay.png')
        panels.append([old,new,full_preview]);results.append(row)
        print(json.dumps(row),flush=True)
    env.close()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    board=Image.new('RGB',(1152,1392),'#171b22');draw=ImageDraw.Draw(board)
    draw.text((12,10),'Case01 | full in-frame regions | external occlusion ignored | review pending',font=font,fill='white')
    for c,title in enumerate(['VISIBLE OBJECT (reference)','FULL OBJECT (new)','FULL GRIPPER + FULL OBJECT']):
        draw.text((c*384+8,46),title,font=font,fill='white')
    for index,(frame,images) in enumerate(zip(FRAMES,panels)):
        y=82+index*434;region=results[index]['regions']['object']
        draw.text((12,y),f'Frame {frame} | object bbox {region["reference_visible_bbox"]} -> {region["bbox"]}',font=font,fill=GREEN)
        for c,img in enumerate(images):board.paste(img.resize((384,384)),(c*384,y+30))
    board.save(out/'overview.png')
    report={'source_match_sha256':digest(OUT/'source_match.json'),'source_excerpt_sha256':digest(EXCERPT),
        'script_sha256':digest(Path(__file__)),'source_key':sample['key'],'frames':FRAMES,
        'region_extent':'full_in_frame','external_occlusion_ignored':True,'screen_bounds_clipped':True,
        'render_each_target_separately':True,'camera_and_pose_adjusted':False,'source_rgb_modified':False,
        'instance_binding':'rule candidates from source Event / task / BDDL, manual review pending',
        'binding_rules_sha256':digest(RULES_PATH),'binding_rules_version':rules['version'],
        'binding_script_sha256':digest(Path(__file__).with_name('bind_instances.py')),'event_bindings':bindings,
        'pose_source':'matched HDF5 full scene states, same as previous visible-object test',
        'visual_geom_names':names,'visual_geom_ids':ids,'results':results,'manual_review':'pending'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fetch','render','render_full'])
    parser.add_argument('--libero-root',type=Path,default=Path('/data1/yuankai/robot-research/workspace/third_party/libero-native'))
    parser.add_argument('--output',type=Path,help='New output directory for render_full; preserve historical previews')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.output and args.stage!='render_full':parser.error('--output is only supported for render_full')
    out=(args.output or OUT.parent/'native_object_full') if args.stage=='render_full' else OUT
    out.mkdir(parents=True,exist_ok=True)
    artifact=out/('source_match.json' if args.stage=='fetch' else 'report.json')
    if artifact.exists():raise FileExistsError(artifact)
    started=time.monotonic()
    if args.stage=='fetch':fetch()
    elif args.stage=='render':render(args.libero_root)
    else:render_full(args.libero_root,out)
    (out/f'{args.stage}_timing.json').write_text(json.dumps({'seconds':time.monotonic()-started})+'\n')


if __name__=='__main__':main()
