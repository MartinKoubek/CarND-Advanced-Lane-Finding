'''
Created on 3. 8. 2017

@author: ppr00076
'''
import cv2

import numpy as np
from moviepy.editor import VideoFileClip
import glob
from camera_calibration import CameraCalibration
from color_gradient_treshold import ColorGradientTreshhold
from perspective_transform import PerspectiveTranform
from detect_lines import DetectLines
from image_show import *
from lane import Lane


camera = CameraCalibration("camera_cal",9,6)
line = DetectLines()
gradient = ColorGradientTreshhold()
transform = PerspectiveTranform()
TEST = 1
frame = 0
FILE = "straight_lines2"

def imageProcessing(img):
    global frame
    frame += 1

    img = np.copy(img)     
    img_calibrated = camera.calibrateImage(img)
#     cv2.imshow("",img)

    cg_img = gradient.pipeline(img_calibrated)
   
    p_img = transform.transform(cg_img) 
    p2_img = transform.transform(img_calibrated) 
#     cv2.imshow("",p_img)
    
#     imageShow(img, \
#               img_calibrated, "Calibrated", \
#               cg_img, "Color/gradient threshold", \
#               p_img, "Perspective transform" )
    
    curve, distance = line.detect(p_img)
    d2_img = line.getLaneDetection()

#     cv2.imshow("",p_img)
    d_img, detected = line.show()
    
    pb_img = transform.transformBack(d_img)
#     cv2.imshow("",pb_img)
    result = cv2.addWeighted(img_calibrated, 1, pb_img, 1, 0)
    
    
    cv2.putText(result, 
                'Radius of Curvature: %.2fm' % curve, 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    
    cv2.putText(result, 'Distance From Center: %.2fm' % (distance), (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if TEST == 1 or detected == False:
        cv2.imwrite("output_images/1_"+ FILE + "_original_image_" + str(frame) + ".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images/2_"+ FILE + "_calibrated_image_" + str(frame) + ".png", cv2.cvtColor(img_calibrated, cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images/3_"+ FILE + "_gradient_image_" + str(frame) + ".png", cg_img)
        cv2.imwrite("output_images/4_"+ FILE + "_transform_image_" + str(frame) + ".png", p_img)
        cv2.imwrite("output_images/4_"+ FILE + "_transform_orig_" + str(frame) + "_image.png", cv2.cvtColor(p2_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images/5_"+ FILE + "_line_detection_" + str(frame) + ".png", cv2.cvtColor(d2_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images/6_"+ FILE + "_line_determination_" + str(frame) + ".png", cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("output_images/7_"+ FILE + "_result.png_" + str(frame) + ".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        imageShow(img, \
                  #img_calibrated, "Calibrated", \
                  cg_img, "Color/gradient threshold", \
                  p_img, "Perspective transform" , \
                  d2_img, "Line detection" , \
                  d_img, "Line determination",  \
                  result, "Result")
        
    return result
        
    
if __name__ == '__main__':
    camera.calibrate()
    
    if (TEST):
#         images = glob.glob('test_images/*.jpg')
#         for img_path in images:
        img = cv2.imread("test_images/" + FILE + ".jpg")   
        img = img[:,:,::-1] 
        imageProcessing(img)  
#             break
    #imageProcessing("test_images/straight_lines1.jpg")
    
    else:
        video_output1 = 'project_video.mp4'
        images_output1 = 'video_test_images/frame%03d.png'
        #images_output1 = 'video_test_images/frame%03d.png'
        video_input1 = VideoFileClip(video_output1)#.subclip(20,27)
        processed_video = video_input1.fl_image(imageProcessing)
        processed_video.write_videofile("out_" + video_output1, audio=False)
        #processed_video.write_images_sequence(images_output1)
        
 
    
    print ("done")
    