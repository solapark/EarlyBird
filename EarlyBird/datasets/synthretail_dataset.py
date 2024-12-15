import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset
import json

intrinsic_camera_matrix_filenames = ['CAM1_intrinsics.xml', 'CAM2_intrinsics.xml', 'CAM3_intrinsics.xml']

class Synthretail(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
        # WILDTRACK has ij-indexing: H*W=480*1440, thus x (i) is \in [0,480), y (j) is \in [0,1440)
        # WILDTRACK has in-consistent unit: centi-meter (cm) for calibration & pos annotation
        self.__name__ = 'Synthretail'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [1000, 1500]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 3, 1880+1861
        # world x,y actually means i,j in Wildtrack, which correspond to h,w
        #self.worldcoord_from_worldgrid_mat = np.array([[0, .1, -90], [.1, 0, -50], [0, 0, 1]])
        self.worldcoord_from_worldgrid_mat = np.array([[.01, 0, -7.5], [0, .01, -5.0], [0, 0, 1]])
        self.intrinsic_matrices = [self.get_intrinsic_matrix(cam) for cam in range(self.num_cam)]
        self.extrinsic_matrices = self.get_extrinsic_matrix()

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 1500
        grid_y = pos // 1500
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        return intrinsic_matrix

    def get_extrinsic_matrix(self):
        with open(os.path.join(self.root, 'calibrations', 'extrinsic.json')) as json_file:
            data = json.load(json_file)

        matrix_list = []
        for key in data:
            inner_matrices = []
            for sub_key in data[key]:
                inner_matrices.append(data[key][sub_key])
            matrix_list.append(inner_matrices)

        return np.array(matrix_list)


