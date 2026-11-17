"""Sparse all-scene bbox audit over every LIBERO40 trajectory, followed by position plots."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from openpi.policies.eskill_dataset import EpisodeReader
from scripts.preprocess.artifacts import digest
from .atomic_vla.bind_instances import resolve_geometry
from .atomic_vla.interaction_targets import scene_entities
from .atomic_vla.scene_regions import mask_box

FIELDS = ["episode_id", "task", "category", "instance", "slot", "frame", "x0", "y0", "x1", "y1", "valid"]
LABELS = {"akita_black_bowl": "黑碗", "alphabet_soup": "字母汤", "basket": "篮子", "bbq_sauce": "烧烤酱",
          "black_book": "黑色书本", "butter": "黄油", "chefmate_8_frypan": "煎锅", "chocolate_pudding": "巧克力布丁",
          "cookies": "饼干", "cream_cheese": "奶油奶酪", "glazed_rim_porcelain_ramekin": "小烤碗",
          "ketchup": "番茄沙司", "milk": "牛奶", "moka_pot": "摩卡壶", "orange_juice": "橙汁", "plate": "盘子",
          "porcelain_mug": "白色马克杯", "red_coffee_mug": "红色咖啡杯", "salad_dressing": "沙拉酱",
          "tomato_sauce": "番茄酱", "white_yellow_mug": "黄白马克杯", "wine_bottle": "酒瓶",
          "desk_caddy": "桌面收纳盒", "flat_stove": "炉灶", "microwave": "微波炉", "white_cabinet": "白色柜子",
          "wine_rack": "酒架", "wooden_cabinet": "木色柜子"}


def extract(config, task_number):
    sys.path.insert(0, config["libero_root"])
    import mujoco
    import robosuite.macros as macros
    from libero.libero.envs import OffScreenRenderEnv

    manifest = json.loads(Path(config["inventory"]).read_text())
    grouped = defaultdict(list)
    for row in manifest["episodes"]:
        grouped[row["bddl"]].append(row)
    output = Path(config["output"])
    (output / "tasks").mkdir(parents=True, exist_ok=True)
    for number, (bddl, rows) in enumerate(grouped.items(), 1):
        if number != task_number:
            continue
        destination = output / "tasks" / (Path(bddl).stem + ".csv")
        if destination.exists():
            raise FileExistsError(destination)
        scene = manifest["scenes"][bddl]
        entities = [v for v in scene_entities(scene).values() if v["kind"] in {"object", "fixture"}]
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=["agentview"], camera_heights=256,
                                camera_widths=256, render_gpu_device_id=config["gpu"], hard_reset=False)
        try:
            env.seed(0)
            env.reset()
            sim, context = env.sim, env.sim._render_context_offscreen
            context.vopt.sitegroup[:] = 0
            context.vopt.geomgroup[:] = 0
            context.vopt.geomgroup[1] = 1
            tracks = []
            for entity in entities:
                geometry = resolve_geometry(env, {**entity, "status": "rule_bound"})
                selected = [sim.model.geom_name2id(name) for name in geometry["visual_geom_names"]]
                tracks.append((entity, selected))
            temporary = destination.with_suffix(".partial.csv")
            with temporary.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                for row in rows:
                    pilot_path = Path(config["pilot_episodes"]) / row["episode_id"] / "episode.json"
                    pilot = EpisodeReader(pilot_path) if pilot_path.exists() else None
                    if digest(row["bddl"]) != row["bddl_sha256"]:
                        raise ValueError("Changed BDDL")
                    match = json.loads(Path(row["source_regions"]).read_text())["scene_match"]
                    if match["status"] != "actions_exact_unique_match" or digest(match["excerpt_path"]) != match["excerpt_sha256"]:
                        raise ValueError("Unmatched or changed source states")
                    with np.load(match["excerpt_path"]) as archive:
                        states = archive["states"]
                    if states.shape != (row["frames"], len(sim.get_state().flatten())):
                        raise ValueError("Scene state dimensions disagree")
                    frames = np.linspace(0, row["frames"]-1, config["frames_per_episode"]).round().astype(int)
                    if len(np.unique(frames)) != len(frames):
                        raise ValueError("Not enough unique source frames")
                    for slot, frame in enumerate(frames):
                        sim.set_state_from_flattened(states[frame])
                        sim.forward()
                        for entity, selected in tracks:
                            context.gl_ctx.make_current()
                            sim.model.geom_group[:] = 5
                            sim.model.geom_group[selected] = 1
                            context.vopt.geomgroup[:] = 0
                            context.vopt.geomgroup[1] = 1
                            seg = sim.render(width=256, height=256, camera_name="agentview", segmentation=True)
                            convention = 1 if macros.IMAGE_CONVENTION == "opengl" else -1
                            seg = seg[::convention][::-1, ::-1]
                            mask = (seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 1], selected)
                            box = mask_box(mask)
                            if pilot is not None:
                                region = pilot.region(pilot.meta["scene_entities"][entity["instance"]]["regions"]["image"], int(frame))
                                expected = None if region is None else region["bbox"]
                                if not np.array_equal(box, expected):
                                    raise ValueError(f'Pilot box mismatch: {row["episode_id"]} / {frame} / {entity["instance"]}: {box} != {expected}')
                            result = dict(episode_id=row["episode_id"], task=Path(bddl).stem, category=entity["category"],
                                          instance=entity["instance"], slot=slot, frame=int(frame), valid=int(box is not None))
                            if box:
                                result.update(zip(("x0", "y0", "x1", "y1"), box))
                            writer.writerow(result)
            temporary.replace(destination)
        finally:
            env.close()
        print(f"TASK {number:02d}/{len(grouped)} | {len(rows)} trajectories | {len(tracks)} objects | {destination.name}", flush=True)


def summarize(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    output = Path(config["output"])
    groups = defaultdict(list)
    all_rows = []
    paths = sorted((output / "tasks").glob("*.csv"))
    if len(paths) != 40 or any(".partial." in p.name for p in paths):
        raise ValueError("All 40 task files must finish before summarizing")
    for path in paths:
        with path.open() as handle:
            all_rows.extend(csv.DictReader(handle))
    if len({r["episode_id"] for r in all_rows}) != 1693 or len({r["task"] for r in all_rows}) != 40:
        raise ValueError("Expected all 1,693 trajectories / 40 tasks")
    for row in all_rows:
        groups[row["category"]].append(row)
    assert set(groups) == set(LABELS)
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    figures = [plt.subplots(7, 4, figsize=(16, 24), constrained_layout=True) for _ in range(2)]
    stats = []
    for i, (category, rows) in enumerate(sorted(groups.items())):
        valid = [r for r in rows if r["valid"] == "1"]
        boxes = np.array([[float(r[k]) for k in ("x0", "y0", "x1", "y1")] for r in valid])
        centers = (boxes[:, :2]+boxes[:, 2:])/2
        initial = centers[np.array([int(r["slot"]) == 0 for r in valid])]
        widths = boxes[:, 2:]-boxes[:, :2]
        quantiles = np.quantile(centers, [.05, .5, .95], axis=0)
        initial_q = np.quantile(initial, [.05, .5, .95], axis=0)
        task_initial_spans = []
        task_summary = []
        for task in sorted({r["task"] for r in rows}):
            selected = np.array([r["task"] == task and int(r["slot"]) == 0 for r in valid])
            if not selected.any():
                continue
            q = np.quantile(centers[selected], [.05, .95], axis=0)
            task_initial_spans.append(q[1]-q[0])
            task_summary.append(dict(task=task, initial_center_p05=q[0].tolist(), initial_center_p95=q[1].tolist()))
        occupancy = np.zeros((256, 256), np.float64)
        for x0, y0, x1, y1 in boxes.astype(int):
            occupancy[y0:y1, x0:x1] += 1
        occupancy /= len(rows)
        bins = np.clip((centers/64).astype(int), 0, 3)
        dominant_bin = np.bincount(bins[:, 1]*4+bins[:, 0], minlength=16).max()/len(centers)
        row = dict(category=category, name=LABELS[category], tasks=len({r["task"] for r in rows}),
                   episodes=len({r["episode_id"] for r in rows}), instance_tracks=len({(r["episode_id"], r["instance"]) for r in rows}),
                   samples=len(rows), valid_samples=len(valid), empty_samples=len(rows)-len(valid),
                   center_p05=quantiles[0].tolist(), center_median=quantiles[1].tolist(), center_p95=quantiles[2].tolist(),
                   center_min=centers.min(0).tolist(), center_max=centers.max(0).tolist(),
                   sampled_bbox_union_xyxy=[float(boxes[:,0].min()), float(boxes[:,1].min()),
                                           float(boxes[:,2].max()), float(boxes[:,3].max())],
                   initial_center_p05=initial_q[0].tolist(), initial_center_p95=initial_q[2].tolist(),
                   initial_range_area_fraction=float(np.prod(initial_q[2]-initial_q[0])/256**2),
                   center_range_area_fraction=float(np.prod(quantiles[2]-quantiles[0])/256**2),
                   dominant_4x4_cell_fraction=float(dominant_bin), median_bbox_wh=np.median(widths,axis=0).tolist(),
                   median_within_task_initial_span_xy=np.median(task_initial_spans,axis=0).tolist(),
                   median_bbox_area_fraction=float(np.median(widths.prod(1))/256**2))
        stats.append(row)
        (output / "per_task").mkdir(exist_ok=True)
        (output / "per_task" / (category+".json")).write_text(json.dumps(task_summary,indent=2)+"\n")
        ax = figures[0][1].flat[i]
        ax.scatter(centers[:, 0], centers[:, 1], s=2, alpha=.12, color="#ed8438", rasterized=True)
        ax.scatter(initial[:, 0], initial[:, 1], s=3, alpha=.3, color="#2076b4", rasterized=True)
        for q, color, ls in ((quantiles, "#d35f00", "-"), (initial_q, "#0966a6", "--")):
            ax.add_patch(Rectangle(q[0], *(q[2]-q[0]), fill=False, edgecolor=color, linewidth=1.2, linestyle=ls))
        heat_ax = figures[1][1].flat[i]
        heat_ax.imshow(occupancy, vmin=0, vmax=1, cmap="magma", extent=(0, 256, 256, 0))
        for plot in (ax, heat_ax):
            plot.set(xlim=(0,256), ylim=(256,0), aspect="equal", xticks=(0,64,128,192,256), yticks=(0,64,128,192,256))
            plot.set_title(f'{LABELS[category]} | {row["tasks"]}任务 / {row["episodes"]}轨迹', fontproperties=font, fontsize=10)
            plot.tick_params(labelsize=7)
            plot.grid(alpha=.15)
    titles = ["主相机 bbox 中心分布：蓝=初始帧；橙=每条轨迹5帧；方框=各轴P5–P95（非联合90%区间）",
              "主相机 bbox 覆盖频率：黑=0；白=100%；每类按含该实例的样本归一化（忽略遮挡）"]
    heat_figure, heat_axes = figures[1]
    colorbar = heat_figure.colorbar(heat_axes.flat[0].images[0], ax=list(heat_axes.flat),
                                  location="bottom", shrink=.55, fraction=.012, pad=.02)
    colorbar.set_ticks([0, .25, .5, .75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    for (figure, axes), title, name in zip(figures, titles, ("centers.png", "bbox_coverage.png")):
        figure.suptitle(title, fontproperties=font, fontsize=15)
        figure.savefig(output/name, dpi=150)
        plt.close(figure)
    result = dict(episodes=1693, tasks=40, category_count=len(stats), sampled_camera_frames=len({(r["episode_id"],r["frame"]) for r in all_rows}),
                  object_observations=len(all_rows), frames_per_episode=config["frames_per_episode"], image_size=[256,256],
                  region_extent="full_in_frame", scope="BDDL objects and finite fixtures, whole entities only",
                  note="Sparse five-frame sampling; extrema are sampled extrema, not full-trajectory bounds. Per-instance observations, not task-balanced.", categories=stats)
    (output/"statistics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    with (output/"statistics.csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=stats[0].keys());writer.writeheader();writer.writerows(stats)
    lines=["# LIBERO40 物品主相机位置分布", "", "全部1,693条轨迹，每条等距抽5帧，共8,465帧；28类物品／固定装置。原图256×256，左上角为(0,0)，x向右、y向下。", "",
           "统计使用完整屏内 bbox，忽略遮挡；部件不另算类别。每个实例每个采样帧权重相同，同类多个实例分别统计。P5–P95分别针对x、y，不能解读成联合90%覆盖；极值只覆盖采样帧。初始位置用每条轨迹第一帧。", "",
           "[中心分布图](centers.png) · [框覆盖频率](bbox_coverage.png) · [完整统计](statistics.csv)", "",
           "| 物品 | 任务数 | 轨迹数 | 中心x P5–P95 | 中心y P5–P95 | 框宽×高 中位数 | 同任务初始跨度x×y 中位数 |", "|---|---:|---:|---|---|---|---|"]
    for r in stats:
        lines.append(f'| {r["name"]} | {r["tasks"]} | {r["episodes"]} | {r["center_p05"][0]:.0f}–{r["center_p95"][0]:.0f} | {r["center_p05"][1]:.0f}–{r["center_p95"][1]:.0f} | {r["median_bbox_wh"][0]:.0f}×{r["median_bbox_wh"][1]:.0f} | {r["median_within_task_initial_span_xy"][0]:.0f}×{r["median_within_task_initial_span_xy"][1]:.0f} |')
    lines.extend(["", "范围面积为各轴P5–P95区间长度之积／整图面积，反映中心位置集中程度；它不是物体占图面积。主相机跨任务布局不同，类别整体分散不保证每个具体场景内也分散。既有仿真状态对齐警示继续适用，本统计用于数据位置偏置核验。"])
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "categories"},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--stage",choices=("extract","summarize"),required=True)
    parser.add_argument("--task-number",type=int,help=argparse.SUPPRESS)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.stage == "summarize":
        summarize(config)
    elif args.task_number is not None:
        extract(config, args.task_number)
    else:
        # Keep each task's native GL resources in its own process. Repeatedly
        # constructing and destroying MuJoCo scenes in one process corrupts rendering.
        manifest=json.loads(Path(config["inventory"]).read_text())
        for number in range(1,len(manifest["scenes"])+1):
            subprocess.run([sys.executable,"-u","-m","scripts.preprocess.libero.object_distribution",
                            "--config",str(args.config),"--stage","extract","--task-number",str(number)],check=True)


if __name__ == "__main__":
    main()
