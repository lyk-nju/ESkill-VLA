"""Render the current LIBERO ESkill model's predictions and attention diagnostics."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Required for offline demonstration probes')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--episode', help='Offline demonstration diagnostic; predicted actions are not executed')
    source.add_argument('--rollout-record', type=Path, help='Real rollout record JSON; follow actual inference requests')
    parser.add_argument('--recording-dir', type=Path, help='Override the recorded server readout directory')
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--attention-video', type=Path, help='Also export the same forward as an attention-only video')
    parser.add_argument('--data-index', type=Path)
    parser.add_argument('--tokenizer-path', type=Path)
    parser.add_argument('--vae-path', type=Path)
    parser.add_argument('--object-word', help='Exact manipulated-object phrase in the instruction (first occurrence)')
    parser.add_argument('--target-word', help='Exact destination phrase in the instruction (last occurrence)')
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--frames', type=int, default=1, help='Keep every Nth source frame')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--fps', type=float, help='Default: recorded control FPS for rollout, 10 for offline probes')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inference-precision', choices=['float32', 'bfloat16'])
    parser.add_argument('--check-action-parity', action='store_true', help='Compare the first frame against unwrapped native inference')
    args = parser.parse_args(argv)
    if args.frames < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('frame interval and frame limit must be positive')
    if args.episode and args.checkpoint is None:
        parser.error('--episode requires --checkpoint')
    if args.rollout_record and (args.frames != 1 or args.max_frames is not None):
        parser.error('rollout rendering uses every recorded control frame; frame sampling is offline-only')
    from scripts.analysis.prediction_viewer import render
    result = render(args.checkpoint, args.episode, args.video, fps=args.fps, device=args.device,
                    attention_video=args.attention_video,
                    rollout_record=args.rollout_record, recording_dir=args.recording_dir,
                    check_action_parity=args.check_action_parity,
                    object_word=args.object_word, target_word=args.target_word,
                    layers=args.layers, stride=args.frames, max_frames=args.max_frames, seed=args.seed,
                    steps=args.steps, data_index=args.data_index, tokenizer_path=args.tokenizer_path,
                    vae_path=args.vae_path, inference_precision=args.inference_precision)
    print(result)


if __name__ == '__main__':
    main()
