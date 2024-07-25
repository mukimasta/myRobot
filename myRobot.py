import logging, sys
import time

import airbot
import pyrealsense2 as rs

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import cv2

from RobotBasics import RobotBasics
from Grasper import Grasper
from Detector import Detector


class myRobot(RobotBasics):
    def __init__(self):
        super().__init__()
        
        self.grasper = Grasper(self)
        self.get_cloud = self.grasper.get_cloud
        self.logger.info("myRobot initialized")
    
        self.detector = None
        self.objects = None
        self.color_image = None
        self.depth_image = None
    
    def set_detector(self, model_path, device="cpu"):
        self.detector = Detector(model_path)
        self.logger.info("Detector set")
        
    def detect_objects(self):
        self.color_image, self.depth_image = self.get_image_and_depth()
        if self.detector is not None:
            result = self.detector.detect_objects(self.color_image)
            self.objects = result
            return result
        else:
            self.logger.error("Detector is not set")
        
    def pick_object(self, id:int):
        if self.objects is not None:
            bbox = self.objects.boxes[id].xyxy[0].to(torch.int).numpy()
            pos = self.grasper.get_grasp(self.color_image, self.depth_image, bbox)
            self.grasper.grasp_at(pos)
        else:
            self.logger.error("Objects are not detected")