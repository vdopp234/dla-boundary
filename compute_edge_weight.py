from imageio import imread
import numpy as np
import os, torch
from cityscapes_single_instance import CityscapesSingleInstanceDataset


def main():
    data_dir = '../cityscapes_dataset'
    num_workers = 8
    train_loader = torch.utils.data.DataLoader(
        CityscapesSingleInstanceDataset(data_dir, 'train', out_dir='edge_weight_outputs/'),
        batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    cnt, bnd_cnt = 0, 0
    for i, (input, target_seg, target_boundary, _) in enumerate(train_loader):
        bnd_cnt += np.sum(target_boundary)
        cnt += np.product(target_boundary.shape)

    f = open("edge_weight_outputs/edge_weight.txt")
    f.write(str(cnt/bnd_cnt))
    f.close()

    print("Edge Weight: ", cnt/bnd_cnt)


if __name__ == '__main__':
    main()