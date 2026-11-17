"""Inspect single-query online augmentation, clean VAE targets and native occlusion."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import textwrap
import time

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

from openpi.training.config import TrainConfig
from openpi.training.data_loader import create_dataset
from openpi.transforms import role_canvas, resize_geometry
from scripts.preprocess.libero.render import font


def record_run(config, out, mode):
    import datetime
    import hashlib
    import os
    import subprocess
    import sys
    files=['src/openpi/augmentation.py','src/openpi/transforms.py','src/openpi/policies/grounding_dataset.py',
           'src/openpi/training/data_loader.py',__file__]
    if Path(sys.argv[0]).is_file():
        files.append(sys.argv[0])
    record=dict(mode=mode,pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),
        command=[sys.executable,*sys.argv],python=sys.executable,seed=config.seed,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True),
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        data_index=config.data_index,split_path=config.split_path,cameras=config.cameras,
        augmentation=asdict(config.grounding_augmentation),vae_path=config.vae_path,
        image_augmentation=asdict(config.image_augmentation),horizontal_flip_prob=config.horizontal_flip_prob,
        smoke=True,wandb_mode='disabled',formal_training=False)
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')


def combined_review(config, out, vae_device=None):
    """One episode per task; use the batch augmentation shared with scripts/train.py."""
    from openpi.augmentation import augment_training_batch
    from openpi.policies.grounding_dataset import collate_grounding
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    record_run(config, out, 'combined augmentation review; no formal training')
    config.save(out/'config.json')
    dataset = create_dataset(config, 'all')
    dataset.augment = True
    selected, tasks = set(), set()
    for ep in range(len(dataset.paths)):
        meta = dataset.reader(ep).meta
        task = meta['instructions'][0]
        if task in tasks:
            continue
        tasks.add(task)
        selected.update((ep,(e['start_frame']+e['stop_frame']-1)//2) for e in meta['events'])
    vae = None
    if vae_device:
        from openpi.models.grounding import load_vae
        vae = load_vae(config.vae_path).to(vae_device)
    def rgb(value):
        return ((value.detach().cpu().permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)
    totals, records, timings = Counter(), [], []
    with av.open(str(out/'review.mp4'), 'w') as video:
        stream=video.add_stream('libx264',rate=5)
        stream.width,stream.height,stream.pix_fmt=1200,1380,'yuv420p'
        stream.options={'crf':'20','preset':'fast','threads':'2'}
        for index,pair in enumerate(dataset.samples):
            if pair not in selected:
                continue
            source=dataset.sample(index,review=True)
            batch=collate_grounding([source])
            torch.manual_seed(config.seed+index)
            start=time.perf_counter()
            final=augment_training_batch(batch,config)
            timings.append(1000*(time.perf_counter()-start))
            row=final['metadata'][0]; info=row['augmentation']
            counts=info.get('counts',{})
            # Task completion keeps the genuine original image and its real Event target.
            assert torch.isfinite(final['inputs']['images']).all()
            assert torch.isfinite(final['targets']['role_images']).all()
            totals['samples']+=1
            totals['mirrored']+=row.get('mirrored',False)
            for name,key in [('cropped','crop_samples'),('rotated','rotated_samples'),('colored','colored_views')]:
                totals[name]+=counts.get(key,0)
            totals['role_valid_before']+=int(batch['targets']['role_valid'].sum())
            totals['role_valid_after']+=int(final['targets']['role_valid'].sum())
            for key in ('copy_applied','target_swapped'):
                totals[key]+=info[key]
            targets=final['targets']['role_images'][0,0]
            recon=[np.zeros((256,256,3),np.uint8) for _ in targets]
            if vae is not None:
                with torch.no_grad():
                    latent=vae.encode(targets.to(vae_device)).latent_dist.mode()
                    decoded=vae.decode(latent).sample.clamp(-1,1)
                assert torch.isfinite(latent).all() and torch.isfinite(decoded).all()
                recon=[rgb(v) for v in decoded]
            panes=[('ORIGINAL MAIN RGB',source['review']['original']),
                   ('AFTER COPY + PATCH',rgb(batch['inputs']['images'][0,0])),
                   ('FINAL MODEL INPUT | geometric + color',rgb(final['inputs']['images'][0,0])),
                   ('CURRENT | before batch transform',rgb(batch['targets']['role_images'][0,0,0])),
                   ('CURRENT | final clean VAE target',rgb(targets[0])),
                   ('CURRENT | FLUX VAE reconstruction' if vae is not None else 'CURRENT | VAE not loaded',recon[0])]
            if len(targets)>1:
                panes += [('SUBJECT | before batch transform',rgb(batch['targets']['role_images'][0,0,1])),
                          ('SUBJECT | final clean VAE target',rgb(targets[1])),
                          ('SUBJECT | FLUX VAE reconstruction' if vae is not None else 'SUBJECT | VAE not loaded',recon[1])]
            sheet=Image.new('RGB',(1200,1380),'#121a20'); draw=ImageDraw.Draw(sheet)
            draw.text((12,8),f'{row["episode_id"]} / frame {row["frame"]}',font=font(19),fill='white')
            for n,line in enumerate(textwrap.wrap(row['task'],105)):
                draw.text((12,34+22*n),line,font=font(18),fill='#ffd98b')
            draw.text((12,82),f'mirror={row.get("mirrored",False)}; rotation={info.get("rotation_degrees",0):.2f} deg; '
                      f'crop side={info.get("crop",[0,0,1,1])[2]:.3f}; color={bool(counts.get("colored_views",0))}; '
                      f'valid={final["targets"]["role_valid"][0,0].tolist()}',font=font(17),fill='white')
            for j,(label,value) in enumerate(panes):
                x,y=10+(j%3)*400,120+(j//3)*415
                for n,line in enumerate(textwrap.wrap(label,38)):
                    draw.text((x,y+n*18),line,font=font(15),fill='white')
                sheet.paste(Image.fromarray(value).resize((365,365)),(x,y+42))
            name=f'{len(records)+1:03d}_{row["episode_id"]}_f{row["frame"]:03d}.jpg'
            sheet.save(out/name,quality=93)
            for _ in range(10):
                for packet in stream.encode(av.VideoFrame.from_image(sheet)):
                    video.mux(packet)
            records.append(dict(**row,preview=name,role_valid=final['targets']['role_valid'][0,0].tolist()))
        for packet in stream.encode():
            video.mux(packet)
    report=dict(tasks=len(tasks),totals=dict(totals),previews=records,
                batch_augmentation_mean_ms=float(np.mean(timings)),batch_augmentation_p95_ms=float(np.percentile(timings,95)),
                timing_scope='CPU batch size 1, excludes Dataset/VAE/rendering; not full training throughput',
                vae_path=config.vae_path if vae else None,vae_device=vae_device,video_seconds=len(records)*2)
    (out/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='previews'}),flush=True)


def marked(rgb, boxes, target, gripper_box=None):
    image = Image.fromarray(rgb).copy()
    draw = ImageDraw.Draw(image)
    for key, box in boxes.items():
        if key != target:
            draw.rectangle(tuple(box), outline='#56cb82', width=1)
    if target in boxes:
        draw.rectangle(tuple(boxes[target]), outline='#ff4343', width=2)
    if gripper_box is not None:
        draw.rectangle(tuple(gripper_box),outline='#48b9ff',width=2)
    return image


def review(config, out, vae_device=None, include=()):
    dataset = create_dataset(config, 'all')
    dataset.augment = True  # This is explicitly a training-augmentation review, never validation loss.
    out.mkdir(parents=True, exist_ok=False)
    record_run(config,out,'review')
    vae = None
    if vae_device:
        from openpi.models.grounding import load_vae
        vae = load_vae(config.vae_path).to(vae_device)
    totals, reasons, previews = Counter(), Counter(), []
    selected = set()
    for ep, path in enumerate(dataset.paths):
        for event in dataset.reader(ep).meta['events']:
            selected.add((ep, (event['start_frame']+event['stop_frame']-1)//2))
    for item in include:
        identity, frame = item.rsplit(':',1)
        ep = next(i for i in range(len(dataset.paths)) if dataset.reader(i).meta['episode_id'] == identity)
        selected.add((ep,int(frame)))
    special = Counter()
    video = av.open(str(out/'review.mp4'), mode='w')
    stream = video.add_stream('libx264', rate=5)
    height = 1380 if config.model.grounding_subject else 960
    stream.width, stream.height, stream.pix_fmt = 1200, height, 'yuv420p'
    started = time.monotonic()
    for index, pair in enumerate(dataset.samples):
        sample = dataset.sample(index, review=True)
        row, data = sample['metadata'], sample['review']
        info = row['augmentation']
        totals['samples'] += 1
        for key in ('copy_attempted','copy_applied','target_swapped','target_moved','augmentation_invalidated',
                    'native_invisible','current_valid'):
            totals[key] += info[key]
        if config.model.grounding_subject:
            totals['subject_valid'] += info['subject_valid']
        reasons[info['copy_reason']] += 1
        # Check every source-resolution canvas, not merely the selected screenshots.
        target = data['target']
        expected, _, _ = role_canvas(data['edited'], {'bbox':data['boxes'][target]} if target else None,
                                     config.model.canvas_size)
        assert np.array_equal(expected, data['canvas'])
        encoded_input = ((sample['targets']['role_images'][0,0].transpose(1,2,0)+1)*127.5).round().astype(np.uint8)
        assert np.array_equal(encoded_input, expected)
        if info['native_invisible']:
            assert not info['augmentation_invalidated'] and info['current_valid']
        if info['target_swapped']:
            reader = dataset.reader(pair[0])
            sequence = reader.meta['scene_entities'][target]['regions'][config.cameras[0]]
            full = reader.region(sequence, row['frame'], include_mask=True)['mask']
            ratio = float(data['visible'].sum() / max(1, full.sum()))
            assert ratio >= config.grounding_augmentation.target_min_visible_fraction
            assert np.isclose(ratio, info['target_visible_fraction'])
            totals['counterfactual_visibility_checks'] += 1
        if config.model.grounding_subject:
            expected_subject, _, available = role_canvas(data['edited'],{'bbox':data['gripper_box']},config.model.canvas_size)
            assert np.array_equal(expected_subject,data['subject_canvas'])
            encoded_subject=((sample['targets']['role_images'][0,1].transpose(1,2,0)+1)*127.5).round().astype(np.uint8)
            assert np.array_equal(encoded_subject,expected_subject)
            assert sample['targets']['role_valid'][0,1] == available
        for protected_box in (data['boxes'].get(target),data['gripper_box']):
            if protected_box is None:
                continue
            margin=config.grounding_augmentation.target_margin_px
            h,w=data['drop'].shape
            box=np.clip(protected_box+[-margin,-margin,margin,margin],[0,0,0,0],[w,h,w,h])
            x0,y0=np.floor(box[:2]).astype(int); x1,y1=np.ceil(box[2:]).astype(int)
            assert not data['drop'][y0:y1,x0:x1].any()
            assert np.array_equal(data['masked'][y0:y1,x0:x1],data['edited'][y0:y1,x0:x1])
        want = pair in selected
        for key in ('target_moved','native_invisible','augmentation_invalidated','copy_applied'):
            if info[key] and special[key] < 3:
                special[key] += 1
                want = True
        if not want:
            continue
        visible = np.zeros_like(data['original'])
        if data['visible'] is not None:
            visible[data['visible']] = (255,255,255)
        reconstructions = [Image.new('RGB',(256,256)) for _ in range(2 if config.model.grounding_subject else 1)]
        latent_shape = None
        if vae is not None:
            value = torch.from_numpy(sample['targets']['role_images'][0]).to(vae_device)
            with torch.no_grad():
                latent = vae.encode(value).latent_dist.mode()
                decoded = vae.decode(latent).sample.clamp(-1,1)
            assert torch.isfinite(latent).all() and torch.isfinite(decoded).all()
            latent_shape = list(latent.shape)
            reconstructions = [Image.fromarray(((rgb.permute(1,2,0).cpu().numpy()+1)*127.5).round().astype(np.uint8))
                               for rgb in decoded]
        grip = data['gripper_box']
        panes = [('ORIGINAL | target red / gripper blue',marked(data['original'],data['original_boxes'],target,grip)),
                 ('COPY-PASTE | full bbox follows object',marked(data['edited'],data['boxes'],target,grip)),
                 ('MODEL INPUT | target + gripper protected',marked(data['masked'],data['boxes'],target,grip))]
        if config.model.grounding_subject:
            from openpi.augmentation import mask_grounding_patches
            _,allowed=mask_grounding_patches(data['edited'],data['interaction_box'],np.random.default_rng(0),
                        replace(config.grounding_augmentation,patch_prob=1,patch_drop_prob=1),config.model.image_size,
                        target_box=data['boxes'].get(target),gripper_box=grip)
            protection=Image.fromarray(np.repeat((~allowed)[:,:,None],3,axis=2).astype(np.uint8)*255)
            panes += [('SUBJECT LABEL | gripper + local RGB',Image.fromarray(data['subject_canvas'])),
                      ('TARGET LABEL | text-selected entity',Image.fromarray(data['canvas'])),
                      ('PROTECT | white patches are never masked',protection),
                      ('SUBJECT LABEL | real FLUX VAE reconstruction',reconstructions[1]),
                      ('TARGET LABEL | real FLUX VAE reconstruction',reconstructions[0]),
                      ('TARGET VISIBLE MASK | not the full bbox',Image.fromarray(visible))]
        else:
            panes += [('TARGET VISIBLE MASK',Image.fromarray(visible)),('TARGET LABEL',Image.fromarray(data['canvas'])),
                      ('REAL FLUX VAE LABEL RECONSTRUCTION',reconstructions[0])]
        canvas = Image.new('RGB',(1200,height),'#121a20'); draw = ImageDraw.Draw(canvas)
        draw.text((12,8),f'{row["episode_id"]} / frame {row["frame"]}',font=font(20),fill='white')
        for line,text in enumerate(textwrap.wrap(row['task'],105)):
            draw.text((12,34+line*22),text,font=font(18),fill='#ffd98b')
        draw.text((12,80),f'copy={info["copy_reason"]}; target moved={info["target_moved"]}; '
                  f'Current valid={info["current_valid"]}; Subject valid={info.get("subject_valid", "n/a")}',font=font(16),fill='white')
        visibility = info['target_visible_fraction']
        visibility_text = f'{visibility:.1%}' if visibility is not None else 'unknown'
        query = 'counterfactual' if info['target_swapped'] else 'real Event'
        draw.text((12,100), f'query={query}; target visible={visibility_text}', font=font(16), fill='#ffd98b')
        for j,(label,im) in enumerate(panes):
            x,y=10+(j%3)*400,120+(j//3)*415
            for line,text in enumerate(textwrap.wrap(label,37)):
                draw.text((x,y+line*18),text,font=font(15),fill='white')
            canvas.paste(im.resize((365,365)),(x,y+42))
        name=f'{len(previews)+1:03d}_{row["episode_id"]}_f{row["frame"]:03d}.jpg'
        canvas.save(out/name,quality=93)
        for _ in range(10):  # Each review sheet is deliberately held for two seconds.
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                video.mux(packet)
        previews.append(dict(**row,preview=name,latent_shape=latent_shape))
    for packet in stream.encode():
        video.mux(packet)
    video.close()
    report=dict(scope='technical data review; no learned prediction or task success claim',
        data_index=config.data_index,augmentation=asdict(config.grounding_augmentation),seed=config.seed,
        episodes=len(dataset.paths),totals=dict(totals),copy_reasons=dict(reasons),
        canvas_checks=totals['samples'],vae_path=config.vae_path if vae else None,vae_device=vae_device,
        previews=previews,seconds=time.monotonic()-started)
    (out/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='previews'}),flush=True)


def export_samples(config, out, count, vae_device=None):
    """Save inspection samples and time CPU preparation, file output and VAE separately."""
    import subprocess
    from openpi.training.config import WandbConfig
    from openpi.training.data_loader import split_episodes

    config = replace(config, smoke=True, wandb_info=WandbConfig(mode='disabled'))
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    record_run(config, out, 'augmentation_export_and_timing')
    config.save(out/'config.json')
    start = time.perf_counter()
    dataset = create_dataset(config, 'all')
    dataset.augment = True
    startup = time.perf_counter()-start
    splits = split_episodes(config.data_index, config.validation_fraction, config.seed, config.split_path)
    membership = {episode: split for split, episodes in splits.items() for episode in episodes}
    # Two independent online draws of each source observation; cover every episode evenly.
    base_count = (count+1)//2
    indices = []
    for ep in range(len(dataset.paths)):
        available = [i for i, (e, _) in enumerate(dataset.samples) if e == ep]
        n = base_count//len(dataset.paths) + (ep < base_count % len(dataset.paths))
        indices.extend(available[i] for i in np.linspace(0, len(available)-1, n, dtype=int))
    keys = [(i, epoch, draw) for draw, i in enumerate(indices) for epoch in range(2)][:count]
    report = dict(scope='technical sample generation; no model training or success-rate evaluation',
                  samples=len(keys), source_frames=len(set(k[0] for k in keys)), episodes=len(dataset.paths),
                  dataset_startup_seconds=startup, cpu_threads=2, preparation=[], export={}, vae=None)
    report['gpu_snapshot'] = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
                                            '--format=csv'], capture_output=True, text=True).stdout
    # Equal keys and alternating order; preparation excludes disk export and VAE.
    for repeat in range(3):
        for variant in (('off', 'on') if repeat % 2 == 0 else ('on', 'off')):
            dataset.augment = variant == 'on'
            for key in keys[:min(64, len(keys))]:
                dataset[key]
            times = []
            for key in keys:
                start = time.perf_counter()
                dataset[key]
                times.append(1000*(time.perf_counter()-start))
            report['preparation'].append(dict(variant=variant, repeat=repeat, samples=len(keys),
                milliseconds=times, mean_ms=float(np.mean(times)), p95_ms=float(np.percentile(times, 95))))
    print(json.dumps(dict(phase='preparation', results=[{k:v for k,v in r.items() if k!='milliseconds'}
                                                       for r in report['preparation']])), flush=True)
    dataset.augment = True
    records, generation_times, write_times, totals = [], [], [], Counter()
    split_counts = Counter()
    for number, key in enumerate(keys):
        start = time.perf_counter()
        sample = dataset.sample(key, review=True)
        generation_times.append(time.perf_counter()-start)
        data, row = sample['review'], sample['metadata']
        destination = out/'samples'/f'{number:04d}'
        start = time.perf_counter()
        destination.mkdir(parents=True)
        # Preserve exact training arrays, including the independent original task-text path.
        arrays = {f'input_{k}':v for k,v in sample['inputs'].items()}
        arrays.update({f'target_{k}':v for k,v in sample['targets'].items()})
        np.savez_compressed(destination/'sample.npz', **arrays)
        input_rgb = ((sample['inputs']['images'][0].transpose(1,2,0)+1)*127.5).round().astype(np.uint8)
        images = dict(original=data['original'], input=input_rgb, current=data['canvas'])
        if config.model.grounding_subject:
            images['subject'] = data['subject_canvas']
        for name, rgb in images.items():
            Image.fromarray(rgb).save(destination/f'{name}.png')
        write_times.append(time.perf_counter()-start)
        # Read back everything; these checks are outside the preparation/write timers.
        with np.load(destination/'sample.npz') as saved:
            for name, value in arrays.items():
                assert np.array_equal(saved[name], value)
        for name, rgb in images.items():
            assert np.array_equal(np.asarray(Image.open(destination/f'{name}.png')), rgb)
        info = row['augmentation']
        for name in ('copy_attempted', 'copy_applied', 'target_swapped', 'target_moved', 'current_valid', 'subject_valid'):
            totals[name] += info.get(name, 0)
        split_counts[membership[row['episode_id']]] += 1
        records.append(dict(**row, sample_key=list(key), split=membership[row['episode_id']],
            source_episode=str(dataset.paths[dataset.samples[key[0]][0]]), path=f'samples/{number:04d}',
            source_shape=list(data['original'].shape),
            original_target_bbox=data['original_boxes'][data['target']].tolist() if data['target'] else None,
            target_bbox=data['boxes'][data['target']].tolist() if data['target'] else None,
            gripper_bbox=data['gripper_box'].tolist() if data['gripper_box'] is not None else None))
    (out/'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    report['export'] = dict(generation_seconds=sum(generation_times), write_seconds=sum(write_times),
                           roundtrip_checked=len(keys), totals=dict(totals), splits=dict(split_counts))
    print(json.dumps(dict(phase='export', **report['export'])), flush=True)
    if vae_device:
        from openpi.models.grounding import load_vae
        def sync():
            if vae_device.startswith('cuda'):
                torch.cuda.synchronize(vae_device)
        sync(); start = time.perf_counter()
        vae = load_vae(config.vae_path).to(vae_device)
        sync(); load_seconds = time.perf_counter()-start
        def role_batch(rows):
            images, owners = [], []
            for i in rows:
                with np.load(out/records[i]['path']/'sample.npz') as saved:
                    valid = saved['target_role_valid']
                    images.extend(saved['target_role_images'][valid])
                    owners.extend((i, int(arm), int(role)) for arm, role in np.argwhere(valid))
            return torch.from_numpy(np.stack(images)), owners
        batch, _ = role_batch(range(min(2, len(keys))))
        with torch.no_grad():
            for _ in range(8):
                vae.encode(batch.to(vae_device)).latent_dist.mode()
        sync()
        times, latents, owners_all = [], [], []
        for offset in range(0, len(keys), 2):
            batch, owners = role_batch(range(offset, min(offset+2, len(keys))))
            sync(); start = time.perf_counter()
            with torch.no_grad():
                latent = vae.encode(batch.to(vae_device)).latent_dist.mode()
                latent = (latent-(getattr(vae.config, 'shift_factor', None) or 0.))*vae.config.scaling_factor
            sync(); times.append(time.perf_counter()-start)
            value = latent.cpu().numpy()
            assert np.isfinite(value).all()
            latents.append(value); owners_all.extend(owners)
        start = time.perf_counter()
        np.savez_compressed(out/'vae_targets.npz', latents=np.concatenate(latents),
                            sample_arm_role=np.asarray(owners_all, np.int64))
        report['vae'] = dict(device=vae_device, weights=config.vae_path, precision='float32',
            observations_per_batch=2, roles=len(owners_all), load_seconds=load_seconds,
            encode_seconds=sum(times), per_observation_ms=sum(times)*1000/len(keys),
            includes_host_to_device=True, write_seconds=time.perf_counter()-start, batch_seconds=times)
    # Compact video built solely from the exported images, two examples per episode.
    video = av.open(str(out/'samples.mp4'), mode='w')
    stream = video.add_stream('libx264', rate=5)
    stream.width, stream.height, stream.pix_fmt = 960, 1040, 'yuv420p'
    selected = []
    for episode in dict.fromkeys(r['episode_id'] for r in records):
        candidates = [i for i,r in enumerate(records) if r['episode_id'] == episode]
        candidates.sort(key=lambda i:(not records[i]['augmentation']['copy_applied'],
                                      not records[i]['augmentation']['target_swapped'], i))
        selected.extend(candidates[:2])
    for i in selected:
        row = records[i]
        sheet = Image.new('RGB', (960,1040), '#121a20'); draw = ImageDraw.Draw(sheet)
        draw.text((12,8),f'{row["episode_id"]} / frame {row["frame"]} / draw {row["sample_key"][1]}',font=font(18),fill='white')
        for line, text in enumerate(textwrap.wrap(row['task'],95)):
            draw.text((12,34+line*22),text,font=font(18),fill='#ffd98b')
        info = row['augmentation']
        visibility = info['target_visible_fraction']
        visibility_text = f'{visibility:.1%}' if visibility is not None else 'unknown'
        draw.text((12,82),f'copy={info["copy_applied"]}; target swap={info["target_swapped"]}; '
                  f'visible={visibility_text}',font=font(18),fill='white')
        names = ['original','input','current'] + (['subject'] if config.model.grounding_subject else [])
        for j, name in enumerate(names):
            x,y = 10+(j%2)*480,115+(j//2)*460
            draw.text((x,y),name.upper()+(' | saved label' if j>=2 else ' | saved RGB'),font=font(18),fill='white')
            with Image.open(out/row['path']/f'{name}.png') as image:
                if j < 2:
                    image = image.copy()
                    h,w = row['source_shape'][:2]
                    nh,nw,top,left = resize_geometry(row['source_shape'],image.width)
                    boxes = [(row['original_target_bbox'] if j==0 else row['target_bbox'], '#ff4343'),
                             (row['gripper_bbox'], '#48b9ff')]
                    overlay = ImageDraw.Draw(image)
                    for box,color in boxes:
                        if box is not None:
                            moved = np.asarray(box)*[nw/w,nh/h,nw/w,nh/h]+[left,top,left,top]
                            overlay.rectangle(tuple(moved),outline=color,width=2)
                sheet.paste(image.resize((430,430)), (x,y+25))
        sheet.save(out/row['path']/'preview.jpg', quality=92)
        for _ in range(10):
            for packet in stream.encode(av.VideoFrame.from_image(sheet)):
                video.mux(packet)
    for packet in stream.encode():
        video.mux(packet)
    video.close()
    report.update(video_samples=selected, bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
    (out/'timing.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(phase='complete', samples=len(keys), video_seconds=2*len(selected),
                         bytes=report['bytes'], vae=report['vae'] and {k:v for k,v in report['vae'].items()
                                                                    if k!='batch_seconds'})), flush=True)


def benchmark(config, out, device):
    """Paired loader + reduced-backbone training timing, using the real frozen VAE."""
    from dataclasses import replace
    from openpi.augmentation import GroundingAugmentation
    from openpi.augmentation import augment_training_batch
    from openpi.models.model import to_device
    from openpi.models.pi0_config import ESkillConfig
    from openpi.training.config import create_model, WandbConfig
    from openpi.training.data_loader import create_data_loader
    import subprocess

    out.mkdir(parents=True,exist_ok=False)
    record_run(config,out,'tiny_training_throughput')
    torch.set_num_threads(2)
    variants={'off':GroundingAugmentation(),'on':config.grounding_augmentation}
    report=dict(scope='technical only: real images/tokenizer/FLUX VAE; random tiny backbone, not full pi05 throughput',
                smoke=True,wandb_mode='disabled',device=device,seed=config.seed,batch_size=2,
                data_index=config.data_index,loader=[],training=[])
    report['gpu_snapshot']=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
                                          '--format=csv'],capture_output=True,text=True).stdout
    for workers in (0,2):
        for repeat in range(2):
            for name in (('off','on') if repeat==0 else ('on','off')):
                cfg=replace(config,device='cpu',num_workers=workers,batch_size=2,grounding_augmentation=variants[name])
                loader=create_data_loader(cfg)
                iterator=iter(loader)
                for _ in range(32):
                    next(iterator)
                start=time.perf_counter()
                for _ in range(128):
                    next(iterator)
                seconds=time.perf_counter()-start
                report['loader'].append(dict(variant=name,workers=workers,repeat=repeat,samples=256,
                                              seconds=seconds,samples_per_second=256/seconds))
                for _ in iterator:
                    pass  # Drain prefetched tensors before closing spawn workers.
                del iterator,loader
    cfg=replace(config,model=ESkillConfig.tiny(visual_primitive_version='v3',freeze_vlm=True,
                    image_size=224,patch_size=14,canvas_size=256,role_queries=16,
                    grounding_subject=config.model.grounding_subject),
                smoke=True,device=device,precision='float32',num_workers=2,batch_size=2,
                pretrained_path=None,wandb_info=WandbConfig(mode='disabled'))
    report['model']=asdict(cfg.model)
    report['vae_path']=cfg.vae_path
    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
    for repeat in range(2):
        for name in (('off','on') if repeat==0 else ('on','off')):
            torch.manual_seed(cfg.seed)
            model=create_model(cfg).train()
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=cfg.learning_rate,weight_decay=cfg.weight_decay,betas=cfg.adam_betas)
            loader=create_data_loader(replace(cfg,grounding_augmentation=variants[name]))
            iterator=iter(loader)
            steps=[]
            for step in range(28):
                sync(); start=time.perf_counter()
                batch=to_device(next(iterator),device)
                batch=augment_training_batch(batch,cfg)
                sync(); loaded=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                losses=model(batch['inputs'],batch['targets'])
                total=sum(cfg.loss_weights[k]*v for k,v in losses.items())
                assert torch.isfinite(total)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
                optimizer.step()
                sync(); end=time.perf_counter()
                if step>=8:
                    steps.append(dict(milliseconds=(end-start)*1000,data_wait_ms=(loaded-start)*1000,
                                      loss=total.item(),current_valid=int(batch['targets']['role_valid'][...,0].sum()),
                                      subject_valid=int(batch['targets']['role_valid'][...,1].sum()) if cfg.model.grounding_subject else 0,
                                      copies=sum(m['augmentation']['copy_applied'] for m in batch['metadata'])))
            report['training'].append(dict(variant=name,repeat=repeat,steps=steps,
                median_ms=float(np.median([s['milliseconds'] for s in steps])),
                mean_ms=float(np.mean([s['milliseconds'] for s in steps])),
                mean_data_wait_ms=float(np.mean([s['data_wait_ms'] for s in steps]))))
            for _ in iterator:
                pass
            del iterator,loader,optimizer,model
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            print(json.dumps({k:v for k,v in report['training'][-1].items() if k!='steps'}),flush=True)
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--vae-device')
    parser.add_argument('--include',action='append',default=[],help='Additional episode_id:frame to display')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--export-samples',type=int,help='Save this many samples and measure preparation/output/VAE separately')
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    config=TrainConfig.load(args.config)
    if args.export_samples is not None:
        if args.export_samples < 1 or args.benchmark:
            parser.error('--export-samples must be positive and cannot be combined with --benchmark')
        if config.horizontal_flip_prob or config.image_augmentation.enabled:
            parser.error('--export-samples exports Dataset-stage arrays only; omit it for combined batch-augmentation review')
        export_samples(config,args.output,args.export_samples,args.vae_device)
    elif args.benchmark:
        benchmark(config,args.output,args.device)
    elif config.horizontal_flip_prob or config.image_augmentation.enabled:
        combined_review(config,args.output,args.vae_device)
    else:
        review(config,args.output,args.vae_device,args.include)


if __name__=='__main__':
    main()
