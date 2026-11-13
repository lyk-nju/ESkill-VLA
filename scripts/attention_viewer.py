"""Render action-query attention over image patches for one or more checkpoints on one episode."""
import argparse
from pathlib import Path


def model_argument(value):
    name, separator, checkpoint = value.partition("=")
    if not separator or not name or not checkpoint:
        raise argparse.ArgumentTypeError("use NAME=CHECKPOINT")
    return name, Path(checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=model_argument, action="append", required=True,
                        help="NAME=CHECKPOINT; repeat to add one video row per model")
    parser.add_argument("--episode", required=True, help="episode_id present in each model's data index")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path,
                        help="episodes index for event/object boxes; default: the first model's data index")
    parser.add_argument("--camera", help="annotated camera drawn in the first view; default: the first model's first camera")
    parser.add_argument("--layers", type=int, nargs="+", help="action-expert layers; default: third quartile")
    parser.add_argument("--frames", type=int, default=1, help="render every Nth source frame")
    parser.add_argument("--seed", type=int, default=0, help="flow noise seed; frame t uses seed + t for every model")
    parser.add_argument("--steps", type=int, default=10, help="flow steps; attention is from the last step")
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--data-index", type=Path, help="Local equivalent of the checkpoint's episode index")
    parser.add_argument("--inference-precision", choices=['float32', 'bfloat16'])
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    from scripts.analysis.attention_viewer import render
    path = render(args.model, args.episode, args.video, annotations=args.annotations, camera=args.camera,
                  device=args.device, layers=args.layers, stride=args.frames, seed=args.seed,
                  steps=args.steps, fps=args.fps, tokenizer_path=args.tokenizer_path,
                  data_index=args.data_index, inference_precision=args.inference_precision)
    print(path)


if __name__ == "__main__":
    main()
