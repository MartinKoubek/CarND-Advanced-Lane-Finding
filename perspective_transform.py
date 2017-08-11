'''
Created on 3. 8. 2017

@author: ppr00076
'''
import numpy as np
import cv2

class PerspectiveTranform(object):


    def __init__(self):
        self.corners_up_left = (600,445)
        self.corners_up_right = (686,445)
        self.corners_dw_left = (280,660)
        self.corners_dw_right = (1038,660)
        
        self.M = None
        
    def transform(self, img):
#         cv2.circle(img, self.corners_up_left, 10, color = ([17, 15, 100]), thickness=10, lineType=1, shift=0) 
#         cv2.circle(img, self.corners_up_right, 10, color=([17, 15, 100]), thickness=10, lineType=1, shift=0) 
#         cv2.circle(img, self.corners_dw_left, 10, color=([17, 15, 100]), thickness=10, lineType=1, shift=0) 
#         cv2.circle(img, self.corners_dw_right, 10, color=([17, 15, 100]), thickness=10, lineType=1, shift=0)  
        
        img_size = (img.shape[1], img.shape[0])
        
        if (self.M == None):
            offset = 300 # offset for dst points
            # Grab the image shape
            
    
            # For source points I'm grabbing the outer four detected corners
            src = np.float32([self.corners_up_left, self.corners_up_right, \
                              self.corners_dw_right, self.corners_dw_left])
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result 
            # again, not exact, but close enough for our purposes
#             dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
#                                          [img_size[0]-offset, img_size[1]-offset], 
#                                          [offset, img_size[1]-offset]])
            dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                        [img_size[0]-offset, img_size[1]], 
                                        [offset, img_size[1]]])
            # Given src and dst points, calculate the perspective transform matrix
            self.M = cv2.getPerspectiveTransform(src, dst)
            self.Mi = cv2.getPerspectiveTransform(dst, src)
            
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, self.M, img_size)

        # Return the resulting image and matrix
        return warped
    
    def transformBack(self, image):
        unwarped = cv2.warpPerspective(image, self.Mi, 
                 dsize = (image.shape[1],image.shape[0]), 
                 flags = cv2.INTER_LINEAR)
        return unwarped
        