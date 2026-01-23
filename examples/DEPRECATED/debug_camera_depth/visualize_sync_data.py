#!/usr/bin/env python
"""
Visualize camera sync test data.

Usage:
    python examples/debug_camera_depth/visualize_sync_data.py debug_camera_output/frame_data.csv
    python examples/debug_camera_depth/visualize_sync_data.py debug_camera_output/frame_data.csv --show-frames 50 55 60
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_sync_data(csv_path: Path, show_frames: list[int] | None = None):
    """Visualize sync test data from CSV."""
    df = pd.read_csv(csv_path)
    output_dir = csv_path.parent

    print(f"Loaded {len(df)} frames from {csv_path}")
    print(f"\nSummary:")
    print(f"  Desync frames: {df['is_desync'].sum()} ({100*df['is_desync'].mean():.1f}%)")
    print(f"  Timestamp diff: mean={df['ts_diff_ms'].mean():.2f}ms, max={df['ts_diff_ms'].max():.2f}ms")
    print(f"  Depth mean range: {df['depth_mean'].min():.0f} - {df['depth_mean'].max():.0f} mm")

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Camera Sync Analysis: {csv_path.name}', fontsize=14)

    # Plot 1: Timestamp difference
    ax1 = axes[0]
    ax1.plot(df['frame_idx'], df['ts_diff_ms'], 'b-', linewidth=0.8, label='ts_diff')
    ax1.axhline(y=33, color='r', linestyle='--', alpha=0.7, label='1 frame threshold (33ms)')
    ax1.fill_between(df['frame_idx'], 0, df['ts_diff_ms'], alpha=0.3)
    ax1.set_ylabel('Timestamp Diff (ms)')
    ax1.set_title('Color/Depth Timestamp Difference')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Highlight desync frames
    desync_frames = df[df['is_desync']]
    if len(desync_frames) > 0:
        ax1.scatter(desync_frames['frame_idx'], desync_frames['ts_diff_ms'],
                   c='red', s=50, zorder=5, label='Desync')

    # Plot 2: Depth mean over time
    ax2 = axes[1]
    ax2.plot(df['frame_idx'], df['depth_mean'], 'g-', linewidth=0.8)
    if 'depth_std' in df.columns:
        ax2.fill_between(df['frame_idx'], df['depth_mean'] - df['depth_std'],
                         df['depth_mean'] + df['depth_std'], alpha=0.2, color='green')
        ax2.set_title('Mean Depth Over Time (with std dev)')
    else:
        ax2.set_title('Mean Depth Over Time')
    ax2.set_ylabel('Depth (mm)')
    ax2.grid(True, alpha=0.3)

    # Highlight large depth changes
    df['depth_change'] = df['depth_mean'].diff().abs()
    large_changes = df[df['depth_change'] > 100]
    if len(large_changes) > 0:
        ax2.scatter(large_changes['frame_idx'], large_changes['depth_mean'],
                   c='red', s=50, zorder=5, marker='x', label=f'Large change (n={len(large_changes)})')
        ax2.legend(loc='upper right')

    # Plot 3: Frame numbers (to check for dropped frames)
    ax3 = axes[2]
    ax3.plot(df['frame_idx'], df['color_frame_num'], 'b-', linewidth=0.8, alpha=0.7, label='Color frame #')
    ax3.plot(df['frame_idx'], df['depth_frame_num'], 'r--', linewidth=0.8, alpha=0.7, label='Depth frame #')
    ax3.set_ylabel('Frame Number')
    ax3.set_title('RealSense Internal Frame Numbers')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Check for frame number gaps
    color_gaps = df['color_frame_num'].diff()
    depth_gaps = df['depth_frame_num'].diff()
    color_dropped = (color_gaps > 1).sum()
    depth_dropped = (depth_gaps > 1).sum()
    if color_dropped > 0 or depth_dropped > 0:
        ax3.set_title(f'RealSense Frame Numbers (color dropped: {color_dropped}, depth dropped: {depth_dropped})')

    # Plot 4: Valid depth percentage (or loop time if not available)
    ax4 = axes[3]
    if 'depth_valid_pct' in df.columns:
        ax4.plot(df['frame_idx'], df['depth_valid_pct'], 'm-', linewidth=0.8)
        ax4.set_ylabel('Valid Depth (%)')
        ax4.set_title('Percentage of Valid Depth Pixels')
        ax4.set_ylim(0, 100)
    elif 'loop_time_ms' in df.columns:
        ax4.plot(df['frame_idx'], df['loop_time_ms'], 'm-', linewidth=0.8)
        ax4.set_ylabel('Loop Time (ms)')
        ax4.set_title('Recording Loop Time')
    ax4.set_xlabel('Frame Index')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / 'sync_analysis.png'
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved analysis plot to: {fig_path}")

    plt.show()

    # Show specific frames if requested
    if show_frames:
        show_frame_images(output_dir, show_frames)

    return df


def show_frame_images(output_dir: Path, frame_indices: list[int]):
    """Show RGB and depth images for specific frames."""
    from PIL import Image

    rgb_dir = output_dir / 'rgb'
    depth_dir = output_dir / 'depth'

    if not rgb_dir.exists() or not depth_dir.exists():
        print("Frame images not found. Run test with --save flag.")
        return

    n_frames = len(frame_indices)
    fig, axes = plt.subplots(2, n_frames, figsize=(4 * n_frames, 6))
    if n_frames == 1:
        axes = axes.reshape(2, 1)

    for i, frame_idx in enumerate(frame_indices):
        rgb_path = rgb_dir / f'frame_{frame_idx:06d}.png'
        depth_path = depth_dir / f'frame_{frame_idx:06d}.png'

        if rgb_path.exists():
            rgb = np.array(Image.open(rgb_path))
            axes[0, i].imshow(rgb)
            axes[0, i].set_title(f'RGB Frame {frame_idx}')
            axes[0, i].axis('off')
        else:
            axes[0, i].text(0.5, 0.5, 'Not found', ha='center', va='center')
            axes[0, i].set_title(f'RGB Frame {frame_idx}')

        if depth_path.exists():
            depth = np.array(Image.open(depth_path))
            # Normalize for visualization
            depth_vis = depth.astype(float)
            depth_vis[depth_vis == 0] = np.nan
            im = axes[1, i].imshow(depth_vis, cmap='viridis', vmin=0, vmax=1000)
            axes[1, i].set_title(f'Depth Frame {frame_idx}\nmean={np.nanmean(depth_vis):.0f}mm')
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04, label='mm')
        else:
            axes[1, i].text(0.5, 0.5, 'Not found', ha='center', va='center')
            axes[1, i].set_title(f'Depth Frame {frame_idx}')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize camera sync test data')
    parser.add_argument('csv_path', type=str, help='Path to frame_data.csv')
    parser.add_argument('--show-frames', type=int, nargs='+', help='Show specific frame images')

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    visualize_sync_data(csv_path, args.show_frames)


if __name__ == '__main__':
    main()
