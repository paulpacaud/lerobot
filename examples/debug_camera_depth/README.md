# Camera Depth Debug Scripts

Scripts to isolate and debug RealSense camera RGB+depth synchronization issues.

## Scripts

### 1. `test_camera_sync.py` - Direct Pipeline Test

Tests the RealSense camera directly without any threading. This isolates whether
the desync is coming from the camera/SDK itself or from our threading code.

```bash
# List available cameras
python examples/debug_camera_depth/test_camera_sync.py --list

# Run 5 second test (default)
python examples/debug_camera_depth/test_camera_sync.py --serial <SERIAL>

# Run 15 second test and save frames for analysis
python examples/debug_camera_depth/test_camera_sync.py --serial <SERIAL> --duration 15 --save

# Full options
python examples/debug_camera_depth/test_camera_sync.py \
    --serial <SERIAL> \
    --duration 10 \
    --fps 30 \
    --width 640 \
    --height 480 \
    --save \
    --output-dir ./my_test_output
```

### 2. `test_async_capture.py` - Async Pattern Test

Tests the background-thread capture pattern used by the robot code. This simulates
how `get_observation()` reads frames during recording.

```bash
# Run async pattern test
python examples/debug_camera_depth/test_async_capture.py --serial <SERIAL>

# With frame saving
python examples/debug_camera_depth/test_async_capture.py --serial <SERIAL> --duration 10 --save
```

## What to Look For

### Timestamp Differences
- `ts_diff_ms < 33ms`: Good synchronization (within 1 frame at 30fps)
- `ts_diff_ms > 33ms`: Potential desync issue

### Depth Changes
- Large sudden changes in depth mean (>100mm between frames) may indicate:
  - Something physically moved in front of the camera
  - OR color/depth are from different time instants (desync)

### Output Analysis
When `--save` is used, the scripts output:
- `rgb/frame_XXXXXX.png` - Color frames
- `depth/frame_XXXXXX.png` - Depth frames (16-bit PNG in mm)
- `frame_data.csv` - Per-frame timing and statistics

You can analyze the CSV to find patterns:
```python
import pandas as pd
df = pd.read_csv('frame_data.csv')

# Find desync frames
desync = df[df['is_desync'] == True]
print(f"Desync frames: {len(desync)}")

# Find large depth changes
df['depth_change'] = df['depth_mean'].diff().abs()
large_changes = df[df['depth_change'] > 100]
print(f"Large depth changes: {len(large_changes)}")
```

## Interpretation

1. **If `test_camera_sync.py` shows desync**: Issue is in RealSense camera/SDK
   - Try different USB port (USB 3.0)
   - Check lighting conditions
   - Update RealSense firmware
   - Reduce resolution/fps

2. **If `test_camera_sync.py` is clean but `test_async_capture.py` shows desync**:
   Issue is in our threading/async code
   - Check frame lock timing
   - May need to adjust capture loop

3. **If both are clean but robot recording shows desync**:
   Issue is in robot recording loop or data saving
   - Robot processing may be causing timing issues
   - Consider profiling the recording loop
