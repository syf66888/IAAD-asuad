import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from nuscenes import NuScenes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import OBS_LEN, TTL_LEN


def compute_gt_trajectories(nusc, scene_name):
    scene = next(scene for scene in nusc.scene if scene["name"] == scene_name)
    curr_sample_token = scene["first_sample_token"]
    last_sample_token = scene["last_sample_token"]
    ego_poses = []

    while True:
        sample = nusc.get("sample", curr_sample_token)
        cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_poses.append(nusc.get("ego_pose", cam_front_data["ego_pose_token"]))
        if curr_sample_token == last_sample_token:
            break
        curr_sample_token = sample["next"]

    ego_traj_world = [pose["translation"][:3] for pose in ego_poses]
    return np.array(ego_traj_world)


def collect_pred_paths(run_dir, scene_name):
    pred_files = {}
    for path in run_dir.glob(f"{scene_name}_*_pred_traj.npy"):
        idx = int(path.stem.split("_")[1])
        pred_files[idx] = path
    return pred_files


def compute_window_ade(gt_future, pred_traj):
    pred_len = min(len(gt_future), len(pred_traj))
    if pred_len == 0:
        return None
    return float(np.mean(np.linalg.norm(gt_future[:pred_len] - pred_traj[:pred_len], axis=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--run-a-dir", required=True)
    parser.add_argument("--run-a-label", default="openemma")
    parser.add_argument("--run-b-dir", required=True)
    parser.add_argument("--run-b-label", default="openemma_plus_note")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    ego_traj_world = compute_gt_trajectories(nusc, args.scene_name)

    run_a_dir = Path(args.run_a_dir)
    run_b_dir = Path(args.run_b_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_a_paths = collect_pred_paths(run_a_dir, args.scene_name)
    run_b_paths = collect_pred_paths(run_b_dir, args.scene_name)
    common_indices = sorted(set(run_a_paths) & set(run_b_paths))

    summary_rows = []
    for idx in common_indices:
        gt_future = ego_traj_world[idx + OBS_LEN: idx + TTL_LEN]
        pred_a = np.load(run_a_paths[idx])
        pred_b = np.load(run_b_paths[idx])

        ade_a = compute_window_ade(gt_future, pred_a)
        ade_b = compute_window_ade(gt_future, pred_b)
        improvement = None if ade_a is None or ade_b is None else ade_a - ade_b

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(gt_future[:, 0], gt_future[:, 1], "r-", linewidth=2, label="GT")
        ax.plot(pred_a[:, 0], pred_a[:, 1], "b-", linewidth=2, label=args.run_a_label)
        ax.plot(pred_b[:, 0], pred_b[:, 1], "g-", linewidth=2, label=args.run_b_label)
        ax.scatter(gt_future[0, 0], gt_future[0, 1], c="k", s=25, label="Start")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{args.scene_name} window {idx}\n"
            f"{args.run_a_label} ADE={ade_a:.3f}, {args.run_b_label} ADE={ade_b:.3f}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{args.scene_name}_{idx}_compare_traj.png")
        plt.close(fig)

        summary_rows.append(
            {
                "scene_name": args.scene_name,
                "window_idx": idx,
                f"{args.run_a_label}_ade": ade_a,
                f"{args.run_b_label}_ade": ade_b,
                "improvement": improvement,
                f"{args.run_a_label}_pred_len": int(len(pred_a)),
                f"{args.run_b_label}_pred_len": int(len(pred_b)),
            }
        )

    summary_path = output_dir / "window_comparison.jsonl"
    with open(summary_path, "w", encoding="utf-8") as handle:
        for row in summary_rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    valid_rows = [row for row in summary_rows if row["improvement"] is not None]
    mean_a = float(np.mean([row[f"{args.run_a_label}_ade"] for row in valid_rows])) if valid_rows else None
    mean_b = float(np.mean([row[f"{args.run_b_label}_ade"] for row in valid_rows])) if valid_rows else None
    mean_improvement = float(np.mean([row["improvement"] for row in valid_rows])) if valid_rows else None

    aggregate = {
        "scene_name": args.scene_name,
        "run_a_label": args.run_a_label,
        "run_b_label": args.run_b_label,
        "num_common_windows": len(common_indices),
        "mean_run_a_ade": mean_a,
        "mean_run_b_ade": mean_b,
        "mean_improvement": mean_improvement,
        "num_windows_improved": sum(1 for row in valid_rows if row["improvement"] > 0),
        "num_windows_degraded": sum(1 for row in valid_rows if row["improvement"] < 0),
        "num_windows_equal": sum(1 for row in valid_rows if row["improvement"] == 0),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
