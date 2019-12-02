from imageio import imread
import numpy as np
import os
from boundary_utils import db_eval_boundary


def main():
    data_path = "../cityscapes_dataset/gtFine/train/"
    cnt, weight = 0, 0
    for folder in os.listdir(data_path):
        curr_path = data_path + folder
        for file in os.listdir(curr_path):
            file_path = curr_path + '/' + file
            if 'instanceId' in file:
                cnt += 1
                img = np.array(imread(file_path))
                boundary_map = db_eval_boundary(img)
                weight += np.sum(boundary_map)
    print("Boundary Weight: ", weight/cnt)


if __name__ == '__main__':
    main()