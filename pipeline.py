import time

import torch
import numpy as np

import Grasper
import Detector

detector = Detector.Detector("./path")
grasper = Grasper.Grasper()

color_image, depth_image = grasper.get_image_and_depth()

hits = detector.detect_object(color_image)
hits.show()

id = input("Which object do you want to grasp: ")

bbox = torch.tensor(hits.boxes[int(id)].xyxy[0], dtype=torch.int).numpy()
print("Bounding Box:", bbox)

pos = grasper.get_grasp(color_image, depth_image, bbox)
grasper.grasp_at(pos)

grasper.move_to(grasper.translations_list, grasper.rotations_list)

