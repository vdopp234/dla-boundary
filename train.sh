CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=9
echo $CUDA_DEVICE_ORDER
echo $CUDA_VISIBLE_DEVICES
python3 segment.py train -d ../cityscapes_dataset/ -c 19 -s 832 --arch dla102up --batch-size 16 --lr 0.01 --momentum 0.9 --lr-mode poly --epochs 20 --random-scale 2 --random-rotate 10 --random-color --pretrained-base imagenet --out-dir ./model_outputs
