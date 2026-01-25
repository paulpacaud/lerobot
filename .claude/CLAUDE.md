# guidelines
- When you write a script, and you need to use a parser, favor typed-argument-parser to define a clean arguments class at the top of the script, it is cleaner.
- When you show an example python usage command, give it inline without line breaks


python examples/post_process_dataset/visualize_ee_with_transform.py --dataset_dir=/home/ppacaud/lerobot_datasets/depth_test_ee --transform_file=ee_transform.json --episode_index=0 --frame_index=100