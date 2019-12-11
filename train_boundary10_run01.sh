python3 segment.py train -d ../cityscapes_dataset/ -c 2 -s 832 --arch dla102up \
	--batch-size 16 --lr 0.005 --momentum 0.9 --lr-mode poly --epochs 50 \
	--random-scale 2 --random-rotate 10 --random-color --pretrained-base imagenet \
	--boundary-detection --edge-weight 10 --out-dir ./model_outputs_boundary_detection10_run01 \
	--wandb

