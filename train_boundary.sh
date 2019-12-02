python3 segment.py train -d ../cityscapes_dataset/ -c 2 -s 832 --arch dla102up \
	--batch-size 16 --lr 0.01 --momentum 0.9 --lr-mode poly --epochs 20 \
	--random-scale 2 --random-rotate 10 --random-color --pretrained-base imagenet \
	--boundary-detection --edge-weight  \
  --out-dir ./model_outputs_boundary_detection
