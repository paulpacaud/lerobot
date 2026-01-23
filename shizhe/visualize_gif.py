"""
Create a GIF from a dataset episode.

Usage:
    python visualize_gif.py --dataset depth_test --episode 0
    python visualize_gif.py --dataset depth_test --episode 0 --fps 10 --max-frames 100
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tap import Tap


SCRIPT_DIR = Path(__file__).parent


class Args(Tap):
    dataset: Path  # Path to dataset directory
    episode: int = 0  # Episode index
    fps: int = 10  # GIF frame rate
    max_frames: int = 150  # Max frames in GIF
    output: Path | None = None  # Output path (default: outputs/episode_X.gif)


def load_video_frames(dataset_path: Path, episode_idx: int) -> list[np.ndarray]:
    """Load RGB frames from video file."""
    import av

    # Find video file
    video_dir = dataset_path / "videos"
    video_keys = [d.name for d in video_dir.iterdir() if d.is_dir() and "depth" not in d.name]

    if not video_keys:
        raise FileNotFoundError(f"No video directories found in {video_dir}")

    video_key = video_keys[0]  # Use first non-depth video
    video_file = list((video_dir / video_key).glob("**/*.mp4"))[0]

    print(f"Loading video from: {video_file}")

    # Get episode frame range from parquet
    parquet_files = sorted((dataset_path / "data").glob("**/*.parquet"))
    episode_start = 0
    episode_length = 0

    for pq_file in parquet_files:
        table = pq.read_table(pq_file)
        df = table.to_pandas()
        ep_df = df[df['episode_index'] == episode_idx]
        if len(ep_df) > 0:
            episode_start = ep_df['index'].min()
            episode_length = len(ep_df)
            break

    if episode_length == 0:
        raise ValueError(f"Episode {episode_idx} not found in dataset")

    print(f"Episode {episode_idx}: frames {episode_start} to {episode_start + episode_length - 1}")

    # Decode video frames
    frames = []
    container = av.open(str(video_file))

    for i, frame in enumerate(container.decode(video=0)):
        if i < episode_start:
            continue
        if i >= episode_start + episode_length:
            break
        frames.append(frame.to_ndarray(format='rgb24'))

    container.close()
    return frames


def create_gif(frames: list[np.ndarray], output_path: Path, fps: int = 10, max_frames: int | None = None):
    """Create GIF from frames."""
    if max_frames and len(frames) > max_frames:
        # Sample frames evenly
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # Convert to PIL images
    pil_frames = [Image.fromarray(f) for f in frames]

    # Save as GIF
    duration = int(1000 / fps)  # ms per frame
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved GIF to: {output_path}")
    print(f"  Frames: {len(pil_frames)}, FPS: {fps}, Duration: {len(pil_frames) / fps:.1f}s")


def main():
    args = Args().parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        return

    # Create output directory
    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = args.output if args.output else output_dir / f"episode_{args.episode}.gif"

    # Load frames
    print(f"Loading episode {args.episode} from {args.dataset}...")
    frames = load_video_frames(args.dataset, args.episode)
    print(f"Loaded {len(frames)} frames")

    # Create GIF
    create_gif(frames, output_path, fps=args.fps, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
