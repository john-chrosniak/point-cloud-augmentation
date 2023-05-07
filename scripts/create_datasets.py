#!/usr/bin/env python3

import os
import numpy as np
import random


def write_dataset():
    # Create perturbed racecar datasets
    for i in range(1,4):
        os.mkdir("../data/racecar/perturbed_{}".format(i))
        for file in os.listdir("../data/racecar/velodyne"):
            points = np.fromfile(os.path.join("../data/racecar/velodyne", file), dtype=np.float32).reshape(-1, 4)
            # Gaussian noise
            points += np.random.normal(0, i*0.01, points.shape)
            # Dropout
            dropout = random.sample(range(len(points)), int(len(points) - len(points) * 0.1*i))
            points = points[dropout]
            with open(os.path.join("../data/racecar/perturbed_{}".format(i), file), 'wb') as pcd_file:
                points.ravel().astype('float32').tofile(pcd_file)
    # Create perturbed kitti datasets
    for i in range(1,4):
        os.mkdir("../data/kitti/perturbed_{}".format(i))
        for file in os.listdir("../data/kitti/velodyne"):
            points = np.fromfile(os.path.join("../data/kitti/velodyne", file), dtype=np.float32).reshape(-1, 4)
            # Gaussian noise
            points += np.random.normal(0, i*0.01, points.shape)
            # Dropout
            dropout = random.sample(range(len(points)), int(len(points) - len(points) * 0.1*i))
            points = points[dropout]
            with open(os.path.join("../data/kitti/perturbed_{}".format(i), file), 'wb') as pcd_file:
                points.ravel().astype('float32').tofile(pcd_file)






if __name__ == "__main__":
    write_dataset()
