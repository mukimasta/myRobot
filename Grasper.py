import logging, sys
import time

import airbot
import pyrealsense2 as rs

import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation

from graspnet import GraspNet

from RobotBasics import RobotBasics

class Grasper:
    def __init__(self, bot: RobotBasics):
        self._bot = bot
        self.grasp_net = GraspNet()
        self.grasp_net.set_factor_depth(1. / self._bot.depth_scale)
        self._bot.logger.info("Grasper initialized")
    
    def get_cloud(self, color_image, depth_image, bbox, margin=20):
        '''
        Get point cloud from the bounding box
        format of bbox: [x1, y1, x2, y2]
        '''
        
        workspace_mask = self.bbox_to_mask(bbox, margin)
        
        end_points, cloud = self.grasp_net.process_data(color_image, depth_image, workspace_mask,
                                                        self._bot.intrinsic_matrix, self._bot.H, self._bot.W)
        
        return end_points, cloud
    
    def get_grasp(self, color_image, depth_image, bbox, show=True):
        '''
        Get grasp by GRASPNET
        '''
        
        end_points, cloud = self.get_cloud(color_image, depth_image, bbox)
        
        grippers = self.grasp_net.get_grasps(end_points, cloud)
        gripper = self.filter_grippers(grippers)
        
        if show:
            # o3d.visualization.draw_geometries([cloud])
            o3d.visualization.draw_geometries([cloud, gripper.to_open3d_geometry()])
        
        gripper = self._bot.cam2base(gripper, offset=0)
        pred_translation, pred_rotation = gripper[:3, 3], gripper[:3, :3]
        rot_euler = Rotation.from_matrix(pred_rotation).as_euler("xyz")
        pred_quat = Rotation.from_matrix(pred_rotation).as_quat()
        pred_translation[2] -= 0.035

        return [pred_translation, pred_quat]
    
    def grasp_at(self, position):
        '''
        Grasp at the given position
        '''
        
        self._bot.grasp = False
        time.sleep(0.5)
        self._bot.move_to(position)
        time.sleep(0.5)
        self._bot.grasp = True
        time.sleep(0.5)


    # ----------------- Utility Functions -----------------
    
    def bbox_to_mask(self, bbox, margin):
        if len(bbox) > 0:
            workspace_mask = np.zeros(shape=[self._bot.H, self._bot.W], dtype=np.bool_)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1 = x1 - margin if x1 - margin > 0 else 0
            y1 = y1 - margin if y1 - margin > 0 else 0
            x2 = x2 + margin if x2 + margin < self._bot.W else self._bot.W
            y2 = y2 + margin if y2 + margin < self._bot.H else self._bot.H       
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            workspace_mask[y1:y2, x1:x2] = True

            return workspace_mask
        else:
            return None
    
    def filter_grippers(self, grippers, angle_threshold=60):
        '''
            Select candidate gripper from cache.
            Select [0, 0, -1] direction grispper
        '''
        angles = []
        # Compute Z direction angle of grisppers
        for gg in grippers:
            # _, _, rotation = self.cam2base(gg, offset=0)
            trans_marix = self._bot.cam2base(gg, offset=0)
            trans, rotation = trans_marix[3, :3], trans_marix[:3, :3]
            z_direction = rotation[:, 0]
            direction = np.array([0, 0, -1])
            dot_product = np.dot(z_direction, direction)
            angle = np.arccos(dot_product)
            angle_degrees = np.degrees(angle) 
            angles.append(angle_degrees)
        angles = np.array(angles)
        # print(angles.shape)
        
        # Filter threshold
        masks = angles < angle_threshold
        angles = angles[masks]
        grippers = grippers[masks]
        
        # Select best gripper
        index = np.argsort(angles)
        # print('index.shape: ', index.shape)
        # print('angles[index[0]]: ', angles[index[0]])
        return grippers[int(index[0])]

    
    
if __name__ == '__main__':
    pass