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

from myRobot import myRobot



class Grasper(myRobot):
    def __init__(self):
        super().__init__()
        self.grasp_net = GraspNet()
        self.grasp_net.set_factor_depth(1. / self.depth_scale)
        self.logger.info("Grasper initialized")
    
    def get_cloud(self, color_image, depth_image, bbox, margin=20):
        '''
        Get point cloud from the bounding box
        format of bbox: [x1, y1, x2, y2]
        '''
        
        workspace_mask = self.bbox_to_mask(bbox, margin)
        
        end_points, cloud = self.grasp_net.process_data(color_image, depth_image, workspace_mask,
                                                        self.intrinsic_matrix, self.H, self.W)
        
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
        
        gripper = self.cam2base(gripper, offset=0)
        pred_translation, pred_rotation = gripper[:3, 3], gripper[:3, :3]
        rot_euler = Rotation.from_matrix(pred_rotation).as_euler("xyz")
        pred_quat = Rotation.from_matrix(pred_rotation).as_quat()
        pred_translation[2] -= 0.035

        return [pred_translation, pred_quat]
        
    def grasp_at(self, position):
        '''
        Grasp at the given position
        '''
        
        translations, rotations = position[0], position[1]
        self.gripper(grasp=False)
        time.sleep(0.5)
        self.move_to(translations, rotations)
        time.sleep(0.5)
        self.gripper(grasp=True)
        time.sleep(0.5)


    # ----------------- Utility Functions -----------------
    
    def bbox_to_mask(self, bbox, margin):
        if len(bbox) > 0:
            workspace_mask = np.zeros(shape=[self.H, self.W], dtype=np.bool_)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1 = x1 - margin if x1 - margin > 0 else 0
            y1 = y1 - margin if y1 - margin > 0 else 0
            x2 = x2 + margin if x2 + margin < self.W else self.W
            y2 = y2 + margin if y2 + margin < self.H else self.H       
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
            trans_marix = self.cam2base(gg, offset=0)
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

    def cam2base(self, gg, offset=0.12):
        '''
            Tranform gripper: Camera --> Object --> End
            Object Frame: Grasped Object
                X: Forward, Y: Right, Z: Down
            Camera Frame: Color-Depth Camera
                X: Right, Y: Down, Z: Forward
            End Frame: 
                X: Down, Y: Left, Z: Forward
        '''
        Tmat_cam2obj = np.eye(4)
        Tmat_cam2obj[:3,:3] = gg.rotation_matrix # @ Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
        Tmat_cam2obj[:3, 3] = gg.translation

        Tmat_base2end = np.eye(4)
        Tmat_base2end[:3,:3] = Rotation.from_quat(self.rotations_list).as_matrix()
        Tmat_base2end[:3, 3] = self.translations_list

        # Static Matrix
        Tmat_end2cam = np.array([
            [0.04855, -0.30770,  0.95024, -0.12346],
            [-0.99602, 0.05626,  0.06910, -0.01329],
            [-0.07472, -0.94982, -0.30375, 0.09521], 
            [0,         0,        0,             1]
        ])

        Tmat_base2obj = Tmat_base2end @ Tmat_end2cam @ Tmat_cam2obj
        rotation = Tmat_base2obj[:3, :3]
        rot_euler = Rotation.from_matrix(rotation).as_euler("xyz")
        if abs(rot_euler[0]) > np.pi / 2.:
            if rot_euler[0] > 0.0:
                rot_euler[0] -= np.pi
            else:
                rot_euler[0] += np.pi

        if (rot_euler[0]*rot_euler[2]) < 0.0:
            if rot_euler[0] > 0.0:
                rot_euler[2] += np.pi
            else:
                rot_euler[2] -= np.pi

        if abs(rot_euler[2]) > np.pi / 2.:
            if rot_euler[2] > 0.0:
                rot_euler[2] -= np.pi
            else:
                rot_euler[2] += np.pi

        
        Tmat_base2obj[:3, :3] = Rotation.from_euler("xyz", rot_euler).as_matrix()
        print(Tmat_base2obj[:3, 3])
        # # Protection Mechanism
        # if Tmat_base2pick_step1[0,3] > 0.40 or Tmat_base2pick_step1[0,3] < 0.05:
        #     print('!!!! ==== X out range ==== !!!!')
        #     Tmat_base2pick_step1[0,3] = min(Tmat_base2pick_step1[0,3], 0.50)
        #     Tmat_base2pick_step1[0,3] = max(Tmat_base2pick_step1[0,3], 0.20)
        # if Tmat_base2pick_step1[1,3] > 0.20 or Tmat_base2pick_step1[1,3] < -0.20:
        #     print('!!!! ==== Y out range ==== !!!!')
        #     Tmat_base2pick_step1[1,3] = min(Tmat_base2pick_step1[1,3],  0.20)
        #     Tmat_base2pick_step1[1,3] = max(Tmat_base2pick_step1[1,3], -0.20)
        if Tmat_base2obj[2, 3] < -0.09:
            print('!!!! ==== Z out range ==== !!!!')
            Tmat_base2obj[2, 3] = max(Tmat_base2obj[2,3], -0.10)

        return Tmat_base2obj
    
    
    
if __name__ == '__main__':
    pass