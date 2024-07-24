import logging, sys
import time

import airbot
import pyrealsense2 as rs

import numpy as np
from scipy.spatial.transform import Rotation
import cv2


class myRobot:
    def __init__(self):
        # set_logger
        self.logger = logging.getLogger("grasp")
        self.logger.setLevel(logging.DEBUG)
        stdoutHandler = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s"
        )
        stdoutHandler.setLevel(logging.DEBUG)
        stdoutHandler.setFormatter(fmt)
        self.logger.addHandler(stdoutHandler)
        self.logger.info("myRobot Logger set up")
        # -----------------------
        self.H, self.W = 720, 1280
        
        self.hardware_init()
        self.gripper(grasp=False)
        
        
    def hardware_init(self):
        # ====================== Initialize Camera =======================
        self.logger.info("Initializing Camera......")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, 6)

        cfg = self.pipeline.start(config)
        depth_sensor = cfg.get_device().query_sensors()[0]
        color_sensor = cfg.get_device().query_sensors()[1]
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        depth_sensor = cfg.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        profile = cfg.get_stream(rs.stream.color)
        intr = profile.as_video_stream_profile().get_intrinsics()
        self.intrinsic_matrix = np.array([[intr.fx, 0.0, intr.ppx],
                                          [0.0, intr.fy, intr.ppy],
                                          [0.0, 0.0, 1.0]])
        print('intrinsic_matrix camera: ', self.intrinsic_matrix, end='\n\n')     
        self.intrinsic_matrix = np.array([[903.481457, 0.0, 647.869078],
                                          [0.0, 908.882758, 350.609108],
                                          [0.0, 0.0, 1.0]])
        print('intrinsic_matrix calibration: ', self.intrinsic_matrix, end='\n\n')

        # ====================== Initialize Robot =======================
        self.logger.info("Initializing Robot......")
        self.translations_list = [0.145, -0.042, 0.2054]
        self.rotations_list = [-0.061, 0.497, -0.059, 0.863]

        
        urdf_path = airbot.AIRBOT_PLAY_WITH_GRIPPER_URDF
        self.bot = airbot.create_agent("down", "can0", 1.0, "gripper", 'OD', 'DM')
        self.bot.set_target_pose(self.translations_list, self.rotations_list, blocking=True)
        
        self.logger.info("Complete!")
    
    def move_from_to(self, now_translations, now_rotations, translations, rotations, step:int=30):
        trans_x_array = np.linspace(now_translations[0], translations[0], step)
        trans_y_array = np.linspace(now_translations[1], translations[1], step)
        trans_z_array = np.linspace(now_translations[2], translations[2], step)
        rota_init_euler = Rotation.from_quat(now_rotations).as_euler('xyz')
        rota_pred_euler = Rotation.from_quat(rotations).as_euler('xyz')
        # print('rota_init_euler,rota_pred_euler: ', rota_init_euler,rota_pred_euler)
        rota_lins_x = np.linspace(rota_init_euler[0], rota_pred_euler[0], step)
        rota_lins_y = np.linspace(rota_init_euler[1], rota_pred_euler[1], step)
        rota_lins_z = np.linspace(rota_init_euler[2], rota_pred_euler[2], step)

        for index in range(len(trans_z_array)-30):
            translation = [trans_x_array[index+30], trans_y_array[index+30], trans_z_array[index+30]]
            rotations = Rotation.from_euler( 'xyz', [rota_lins_x[index+30], rota_lins_y[index+30], rota_lins_z[index+30]]).as_quat()
            self.bot.set_target_pose(translation, rotations, blocking=False, use_planning=True)
            time.sleep(1/step)
        self.bot.set_target_pose(translations, rotations, blocking=True, use_planning=True)
    
    def move_by(self, translations, rotations, step:int=30):
        self.logger.info("Start Moving by: " + str(translations) + str(rotations))
        
        now_translations, now_rotations = np.zeros(3), np.zeros(4)
        self.move_from_to(now_translations, now_rotations, translations, rotations, step)
        
        now_translations, now_rotations = self.bot.get_current_pose()
        self.logger.info("Complete Moving to: " + str(now_translations) + str(now_rotations))
    
    def move_to(self, translations, rotations):
        now_translations, now_rotations = self.bot.get_current_pose()
        self.logger.info("Now Position: " + str(now_translations) + str(now_rotations))
        
        self.move_from_to(now_translations, now_rotations, translations, rotations)
        self.logger.info("Complete Moving to: " + str(translations) + str(rotations))

    def gripper(self, grasp: bool):
        if grasp:
            self.bot.set_target_end(0)
        else:
            self.bot.set_target_end(1)
            
        self.logger.info("Gripper Grasping: " + str(grasp))
    
    def get_image_and_depth(self):
        '''
            Read color and depth image from stream.
            Depth is align to color image.
        '''
        time.sleep(1)
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.array(depth_frame.get_data()).astype(np.float32)
        color_image = np.array(color_frame.get_data())
        # depth_imagemap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imwrite('color.png', color_image)
        # cv2.imwrite('depth.png', depth_imagemap)
        # img_origin = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # BGR to RGB
        
        return color_image, depth_image
    

if __name__ == "__main__":
    bot = myRobot()
    bot.gripper(grasp=False)
    time.sleep(1)
    bot.gripper(grasp=True)
    