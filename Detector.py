from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch

class Detector:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        
        self.model = YOLO(model_path)
        self.model = self.model.to(device)
        
        # read file with classes from "default_classes.txt devided by comma"
        self.classes = []
        try:
            # with open("RobotGraspTest/default_classes.txt", "r") as file:
            with open(os.path.join("RobotGraspTest", "default_classes.txt"), "r") as file:
                self.classes = file.read().split(", ")
        except:
            with open("default_classes.txt", "r") as file:
                self.classes = file.read().split(", ")
        
        self.model.set_classes(self.classes)
        
    def set_classes(self, classes):
        self.classes = classes
        self.detector.set_classes(classes)
    
    def detect_object(self, color_image):
        '''
        Detect object in the image
        '''
        results = self.model.predict(color_image)
        
        if self.device != "cpu":
            result = results[0].cpu()
        else:
            result = results[0]
            
        result.boxes.is_track = True
        result.boxes.data = torch.cat(
            (result.boxes.data[:, :4], torch.arange(len(result)).unsqueeze(1), result.boxes.data[:, 4:]), 
            dim=1
        )
            
        return result
    

if __name__ == '__main__':
    detector = Detector("yolov8s-worldv2.pt")
    # print(detector.classes)
    color_image = cv2.imread("test/test3.jpg")
    
    hits = detector.detect_object(color_image)
    hits.show()