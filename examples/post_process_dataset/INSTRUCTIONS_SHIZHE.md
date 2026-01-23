Pull the dataset:
```
huggingface-cli download paulpacaud/depth_test_v2 \
  --repo-type dataset \
  --local-dir depth_test \
  --local-dir-use-symlinks False
```

Download my visualization script in the folder. The folder should look like:

Visualize the data:
```
python ./visualize_postprocessed_pcd.py \
    --dataset_dir=./depth_test_v2 \
    --episode_index=0 --frame_index=100
```