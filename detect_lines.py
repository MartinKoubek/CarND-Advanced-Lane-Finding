'''
Created on 2. 8. 2017

@author: ppr00076
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import dtype
from builtins import int

FILTER = 3

class DetectLines():
    def __init__(self):
        self.reset()
        self.my_average_polyfit_left = []        
        self.my_average_polyfit_right = []
        
    def reset(self):
        self.my_left_lane = []
        self.my_left_lane_x = []
        self.my_left_lane_y = []
        
        self.my_right_lane = []
        self.my_right_lane_x = []
        self.my_right_lane_y = []     

        self.end_y = None
    
    def detect(self, binary_warped): 
        self.reset()
        self.binary_warped = np.copy(binary_warped) 
        self.binary_warped_out = np.copy(binary_warped) 
        self.binary_warped_out = cv2.cvtColor(self.binary_warped, cv2.COLOR_GRAY2BGR)
        self.binary_warped_out = cv2.cvtColor(self.binary_warped_out, cv2.COLOR_BGR2RGB)
        #binary_warped = binary_warped.astype(int)
        # Create an output image to draw on and  visualize the result

        out_img = (np.dstack((self.binary_warped, self.binary_warped, \
                              self.binary_warped))*255).astype('uint8')
        
 
        # Choose the number of sliding windows
        nwindows = 9
        
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        
        # Set height of windows
        window_width = 50
        
        window_height = np.int(self.binary_warped.shape[0]/nwindows)
        
        window = np.ones(window_width)
        
        self.left_fit2 = None
        self.right_fit2 = None
        
        if (True):
            # Assuming you have created a warped binary image called "self.binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(self.binary_warped[450:,:], axis=0)
            
            plt.imshow(self.binary_warped)
            plt.plot(histogram)
            
            
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[10:midpoint])
            rightx_base = np.argmax(histogram[midpoint:-10]) + midpoint
        
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = self.binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            
#             self.my_left_lane_x.extend([leftx_current])
#             self.my_left_lane_y.extend([720])
#             self.my_left_lane.extend([[leftx_current,720]])
#             
#             self.my_right_lane_x.extend([rightx_current])
#             self.my_right_lane_y.extend([720])
#             self.my_right_lane.extend([[rightx_current,720]])
                    
            
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []
            
            
            
            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
                win_y_high = self.binary_warped.shape[0] - window*window_height
                win_xleft_low = int((leftx_current - margin)/1)
                win_xleft_high = int((leftx_current + margin)/1)
                win_xright_low = int((rightx_current - margin)/1)
                win_xright_high = int((rightx_current + margin)/1)
                
                # Draw the windows on the visualization image
                cv2.rectangle(self.binary_warped_out,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(10,100,100), 5) 
                cv2.rectangle(self.binary_warped_out,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(10,100,100), 5) 
                
                # Identify the nonzero pixels in x and y within the window
#                 plt.imshow(self.binary_warped)
                
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                left_dx_current = 0
                right_dx_current = 0
                found_l = True
                found_r = True
                if len(good_left_inds) > minpix:
                    left_dx_current = leftx_current;
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                    lefty_current = np.int(np.mean(nonzeroy[good_left_inds]))
                    
                    left_dx_current = left_dx_current - leftx_current;
                    
                    self.my_left_lane_x.extend([leftx_current])
                    self.my_left_lane_y.extend([lefty_current])
                    self.my_left_lane.extend([[leftx_current,lefty_current]])
                    
                    #self.my_left_lane_x.extend([leftx_current])
                    #self.my_left_lane_y.extend([win_y_high])
                else:
                    found_l = False
                    
                if len(good_right_inds) > minpix:     
                    right_dx_current =  rightx_current; 
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                    righty_current = np.int(np.mean(nonzeroy[good_right_inds]))
                    
                    right_dx_current = right_dx_current - rightx_current
                    
                    self.my_right_lane_x.extend([rightx_current])
                    self.my_right_lane_y.extend([righty_current])
                    self.my_right_lane.extend([[rightx_current,righty_current]])
                    
                else:
                    rightx_current -= left_dx_current
                    found_r = False
                
                if found_l == False and found_r == False:
                    self.end_y = win_y_low
                    break;
                    
                elif found_l == False:
                   leftx_current -= right_dx_current 
#             Concatenate the arrays of indices

            for xy in (self.my_left_lane):
                cv2.circle(self.binary_warped_out,(xy[0], xy[1]),10,(255,72,35), 3)
            for xy in (self.my_right_lane):
                cv2.circle(self.binary_warped_out,(xy[0], xy[1]),10,(167,255,45), 3) 
                

            if (len(self.my_left_lane_y) > 1):
                left_fit2 = np.polyfit(self.my_left_lane_y, 
                                            self.my_left_lane_x, deg=2)
                self.my_average_polyfit_left.append(left_fit2)
                if (len(self.my_average_polyfit_left)) > FILTER:
                    self.my_average_polyfit_left.pop(0)
            else:
                if (len(self.my_average_polyfit_left)) > 0:
                    self.my_average_polyfit_left.pop(0)
            
            if (len(self.my_right_lane_y) > 1):
                right_fit2 = np.polyfit(self.my_right_lane_y, 
                                            self.my_right_lane_x, deg=2)
                self.my_average_polyfit_right.append(right_fit2)
                
            
                if (len(self.my_average_polyfit_right)) > FILTER:
                    self.my_average_polyfit_right.pop(0)
            else:
                if (len(self.my_average_polyfit_right)) > 0:
                    self.my_average_polyfit_right.pop(0)
                    
            
            ym_per_pix = 3.048/275
            ym_per_pix = 30./720
            
            xm_per_pix = 3.7/413
            xm_per_pix = 3.7/700


            # Calculate the new radii of curvature
            h = self.binary_warped.shape[0]
            self.ploty = np.linspace(0, h-1, h)
            y_eval = np.max(self.ploty)
            left_curverad = 0
            right_curverad = 0
            
            left_center = 0
            right_center = 0
            
            if (len(self.my_left_lane_y) > 0 and 
                len(self.my_left_lane_x) > 0) and len(self.my_average_polyfit_left) > 0:
                a = np.array(self.my_left_lane_y)*ym_per_pix
                b = np.array(self.my_left_lane_x)*xm_per_pix
                self.left_fit = np.polyfit(a, b, deg=2)
                left_curverad = ((1 + (2*self.left_fit[0]*y_eval*ym_per_pix + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
                left_center = int(np.polyval(self.my_average_polyfit_left[0], 720))
            
            if (len(self.my_right_lane_y) > 0 and 
                len(self.my_right_lane_x) > 0)and len(self.my_average_polyfit_right) > 0:
                a = np.array(self.my_right_lane_y)*ym_per_pix
                b = np.array(self.my_right_lane_x)*xm_per_pix
                self.right_fit = np.polyfit(a,b,deg=2)

                right_curverad = ((1 + (2*self.right_fit[0]*y_eval*ym_per_pix + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])        
                right_center = int(np.polyval(self.my_average_polyfit_right[0], 720))
        
            lane_center_position = midpoint - (right_center + left_center)/2
            lane_center_position *= xm_per_pix 
            
            
            self.binary_warped = cv2.cvtColor(self.binary_warped, cv2.COLOR_GRAY2BGR)
            return int((left_curverad+right_curverad)/2), lane_center_position

    def getLaneDetection(self):
        return self.binary_warped_out
    
    def show(self): 
        warp_zero = np.zeros_like(self.binary_warped).astype(np.uint8)
        #warp_zero = cv2.cvtColor(warp_zero, cv2.COLOR_GRAY2BGR)
        
        if self.my_average_polyfit_left == None or self.my_average_polyfit_right == None:
            return warp_zero, False
        
        if len(self.my_average_polyfit_left) == 0 or len(self.my_average_polyfit_right) == 0:
            return warp_zero, False
        
        x = np.array(self.my_left_lane)
        y = np.array(self.my_right_lane)
        
        x1 = np.array([], dtype = np.int32)
        y1 = np.array([], dtype = np.int32)
        left_xy = []
        right_xy = []
        
        # Average polyfit parameters
        left_fit2 = [0]*3
        for i,val in enumerate(self.my_average_polyfit_left):
            left_fit2[0] += val[0]
            left_fit2[1] += val[1]
            left_fit2[2] += val[2]
        
        left_fit2[0] = (left_fit2[0]/len(self.my_average_polyfit_left))
        left_fit2[1] = (left_fit2[1]/len(self.my_average_polyfit_left))
        left_fit2[2] = (left_fit2[2]/len(self.my_average_polyfit_left))
        
        right_fit2 = [0]*3
        for i,val in enumerate(self.my_average_polyfit_right):
            right_fit2[0] += val[0]
            right_fit2[1] += val[1]
            right_fit2[2] += val[2]
        
        right_fit2[0] = (right_fit2[0]/len(self.my_average_polyfit_right))
        right_fit2[1] = (right_fit2[1]/len(self.my_average_polyfit_right))
        right_fit2[2] = (right_fit2[2]/len(self.my_average_polyfit_right))
        
        #get x,y from polynom
        for i in range (720,-1, -100):
            if self.end_y != None:
                if self.end_y >= i:
                    break;
            
            left_xy.extend([[int(np.polyval(left_fit2, i)),i]])
            right_xy.extend([[int(np.polyval(right_fit2, i)),i]])
            #xx = np.polyval(self.left_fit, i,)
            #x1 = np.append(x1,int(xx))
            #y1 = np.append(y1,i)
            #a = np.concatenate(a, np.array([[int(xx)],[i]],dtype = np.int32))
         
         
        left_xy = np.array(left_xy, dtype = np.int32)
        right_xy = np.array(right_xy, dtype = np.int32)
        
        #write lines to image
        pts = np.array(np.concatenate((left_xy,right_xy[::-1]), axis=0),dtype=np.int32)
        #print (pts)
        cv2.fillPoly(warp_zero, [pts], color = (0,255, 0))
        cv2.polylines(warp_zero, [left_xy], isClosed = False,color = (255,0,0), thickness = 50)
        cv2.polylines(warp_zero, [right_xy],  isClosed =False, color = (255,0,255), thickness = 50)
                      
        #cv2.imshow("",warp_zero)
#         plt.plot(self.left_fit,  color='yellow')
#         plt.plot(self.right_fit,  color='yellow')
#         plt.xlim(0, 1280)
#         plt.ylim(720, 0)
        return warp_zero, True
    
if __name__ == '__main__':
    f = FindingLines()
    f.preProcessImage("warped-example.jpg")