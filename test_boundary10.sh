python3 segment.py test -d ../cityscapes_dataset/ -c 2 -s 832 --arch dla102up \
	--batch-size 16 --lr 0.01 --momentum 0.9 --lr-mode poly --epochs 20 \
	--random-scale 2 --random-rotate 10 --random-color --pretrained-base imagenet \
	--boundary-detection --edge-weight 10 --out-dir ./model_outputs_boundary_detection10 \
	--resume ./model_outputs_boundary_detection10/checkpoint_latest.pth

