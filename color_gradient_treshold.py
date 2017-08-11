'''
Created on 3. 8. 2017

@author: ppr00076
'''
import numpy as np
import cv2
from image_show import *

class ColorGradientTreshhold(object):

    def __init__(self):
        pass
        
    def abs_sobel_thresh(self,gray, orient='x', sobel_kernel = 3,  thresh=(0,255)):
        # Apply the following steps to img
        # 1) Convert to grayscale

        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        elif orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        
        # 3) Take the absolute value of the derivative or gradient
        
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
#         cv2.imshow("sxbinary", sxbinary)

        # 6) Return this mask as your binary_output image
        #binary_output = np.copy(img) # Remove this line
        
        return sxbinary
    
    def mag_thresh(self,gray, sobel_kernel=3, thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale

        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # 3) Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max

        sxbinary = np.zeros_like(gradmag)
        sxbinary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        #binary_output = np.copy(img) # Remove this line
        
        return sxbinary

    def dir_threshold(self,gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    

        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # 3) Calculate the gradient magnitude
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max

        sxbinary = np.zeros_like(absgraddir)
        sxbinary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        #binary_output = np.copy(img) # Remove this line
        
        return sxbinary
    
    def hls_select(self,img, thresh=(0, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        # 2) Apply a threshold to the S channel
        S = hls[:,:,1]
        s_hls = np.zeros_like(S)
        s_hls[(S > thresh[0]) & (S <= thresh[1])] = 1
        
        ## White Color
        lower_white = np.array([0, 210, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)
    
        ## Yellow Color
        lower_yellow = np.array([18, 0, 100], dtype=np.uint8)
        upper_yellow = np.array([30, 220, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    
        combined_binary = np.zeros_like(white_mask)
        combined_binary[((white_mask == 255) | (yellow_mask == 255))] = 255

        
        
        
        # 3) Return a binary image of threshold result       
        return combined_binary
    
    def hls_select_L(self,img, thresh=(0, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        # 2) Apply a threshold to the S channel
        L = hls[:,:,1]
        l_hls = L*(255/np.max(L))
        
        binary_output = np.zeros_like(l_hls)
        binary_output[(l_hls > thresh[0]) & (l_hls <= thresh[1])] = 1

        # 3) Return a binary image of threshold result       
        return binary_output
        
    def lab(self,img):
        # 1) Convert to LAB color space
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         cv2.imshow("bgr",bgr )
        s_channel = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)[:,:,2]
#         cv2.imwrite("3_s_channel_image.png", s_channel)
        l_channel = cv2.cvtColor(bgr, cv2.COLOR_BGR2LUV)[:,:,0]
#         cv2.imwrite("3_l_channel_image.png", l_channel)
        b_channel = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[:,:,2]
#         cv2.imwrite("3_b_channel_image.png", b_channel)
                
        s_thresh_min = 180
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 255
#         cv2.imwrite("3_s_treshold_channel_image.png", s_binary)

        
        b_thresh_min = 155
        b_thresh_max = 200
        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 255
#         cv2.imwrite("3_b_treshold_channel_image.png", b_binary)
        
        l_thresh_min = 225
        l_thresh_max = 255
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 255
#         cv2.imwrite("3_l_treshold_channel_image.png", l_binary)
    
        #color_binary = np.dstack((u_binary, s_binary, l_binary))
        
        combined_binary = np.zeros_like(s_binary)
        combined_binary[(l_binary == 255) | (s_binary == 255)] = 255
#         cv2.imshow("combined_binary",combined_binary )
        
        return combined_binary
    
    def pipeline(self,img):
        
        ksize = 3 
        img2 = np.copy(img)
        gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(30, 230))
        grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 230))
        mag_binary = self.mag_thresh(gray, sobel_kernel=ksize, thresh=(80, 150))
        dir_binary = self.dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
        hls_binary = self.hls_select(img2, thresh=(220, 255))
        lab_binary = self.lab(img2)
#         cv2.imshow("hls_binary", hls_binary)
        
        #combined = np.zeros_like(dir_binary)
        #combined[((gradx == 1) | (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        
        combined = np.zeros_like(lab_binary)
        combined[((gradx == 1) | (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        #combined[(hls_binary == 1) | (lab_binary == 1)] = 1
        
        #combined[((gradx == 1) & (grady == 1)) | (rbinary == 1)] = 1
        
#         imageShow(img2, \
#                   mag_binary, "mag_binary", \
#                   gradx, "gradx", \
#                   hls_binary, "hls_binary" , \
#                   dir_binary, "dir_binary",  \
#                   combined, "combined")
        

#         img = np.copy(img)
# #         # Convert to HSV color space and separate the V channel
# #         hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
# #         l_channel = hsv[:,:,1]
# #         s_channel = hsv[:,:,2]
# #         # Sobel x
#         sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
#         abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
#         scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
#         
#         # Threshold x gradient
#         sxbinary = np.zeros_like(scaled_sobel)
#         sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
#         
#         # Threshold color channel
#         s_binary = np.zeros_like(s_channel)
#         s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
#         # Stack each channel
#         # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
#         # be beneficial to replace this channel with something else.
#         color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
#         
#         combined = np.zeros_like(dir_binary)
#         combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        kernel = np.ones((5, 5), np.uint8)  
        closing = cv2.morphologyEx(hls_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return closing 