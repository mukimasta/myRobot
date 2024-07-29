import numpy as np
import cv2
import time

import pyrealsense2 as rs

W, H = 1280, 720



def init_camera():
    # ====================== Initialize Camera =======================
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 6)

    cfg = pipeline.start(config)
    depth_sensor = cfg.get_device().query_sensors()[0]
    color_sensor = cfg.get_device().query_sensors()[1]
    depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    depth_sensor = cfg.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()
    intrinsic_matrix = np.array([[intr.fx, 0.0, intr.ppx],
                                        [0.0, intr.fy, intr.ppy],
                                        [0.0, 0.0, 1.0]])
    print('intrinsic_matrix camera: ', intrinsic_matrix, end='\n\n')     
    intrinsic_matrix = np.array([[903.481457, 0.0, 647.869078],
                                        [0.0, 908.882758, 350.609108],
                                        [0.0, 0.0, 1.0]])
    print('intrinsic_matrix calibration: ', intrinsic_matrix, end='\n\n')

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

if __name__ == '__main__':
    init_camera()
    
    color_image, depth_image = get_image_and_depth()
    
    cv2.imshow("color", color_image)

