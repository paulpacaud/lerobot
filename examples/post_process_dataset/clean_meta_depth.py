import json
import jsonlines
from pathlib import Path

FEATURES_TO_REMOVE = ["observation.images.front_depth"]
DATASETS = [
  Path("/home/ppacaud/lerobot_datasets/put_sockets_into_drawer_pointact"),
  Path("/home/ppacaud/lerobot_datasets/put_banana_and_toy_in_plates_pointact"),
  Path("/home/ppacaud/lerobot_datasets/put_cube_in_spot_pointact"),
]

for dataset_path in DATASETS:
  # Clean stats.json
  stats_file = dataset_path / "meta" / "stats.json"
  with open(stats_file) as f:
      stats = json.load(f)
  for feature in FEATURES_TO_REMOVE:
      stats.pop(feature, None)
  with open(stats_file, "w") as f:
      json.dump(stats, f, indent=2)

  # Clean episodes_stats.jsonl
  episodes_file = dataset_path / "meta" / "episodes_stats.jsonl"
  with jsonlines.open(episodes_file) as reader:
      lines = list(reader)
  for entry in lines:
      for feature in FEATURES_TO_REMOVE:
          entry.get("stats", {}).pop(feature, None)
  with jsonlines.open(episodes_file, mode="w") as writer:
      writer.write_all(lines)

  print(f"Cleaned: {dataset_path.name}")