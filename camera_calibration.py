'''
Created on 2. 8. 2017

@author: ppr00076
'''
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

class CameraCalibration():
    def __init__(self, path, nx, ny):
        self.path = path
        self.nx = nx
        self.ny = ny
    
        dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]
    
    def calibrate(self):
        
        if (self.mtx != None):
            return
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.nx*self.ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        
        # Make a list of calibration images
        images = glob.glob(self.path+'/calibration*.jpg')
        
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)
        
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (self.nx,self.ny), corners, ret)
                #write_name = 'corners_found'+str(idx)+'.jpg'
                #cv2.imwrite(write_name, img)
                #cv2.imshow('img', img)
                #cv2.waitKey(500)
            else:
                print("can not find chessboard: " + fname)
        
        #cv2.destroyAllWindows()
        
        img_size = (img.shape[1], img.shape[0])        
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        
#         for idx, fname in enumerate(images):
#             img = cv2.imread(fname)
#             ret, warped, self.M = self.getUnwarpParameters(img)
#             if ret == True:
#                 cv2.imwrite("0_camera_calibration_" + str(idx) + ".png", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
#         
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump(dist_pickle, open( "wide_dist_pickle.p", "wb" ) )
        return
        
        
    def calibrateImage(self, img):               
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        img_size = (img.shape[1], img.shape[0])  
        #warped = cv2.warpPerspective(undist, self.M, img_size)
        return undist

    def getUnwarpParameters(self, img):
        # Use the OpenCV undistort() function to remove distortion
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        #cv2.imshow('undist', undist)
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        # Search for corners in the grayscaled image
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
    
        if ret == True:
            # If we found corners, draw them! (just for fun)
            #cv2.drawChessboardCorners(undist, (self.nx, self.ny), corners, ret)
            # Choose offset from image corners to plot detected corners
            # This should be chosen to present the result at the proper aspect ratio
            # My choice of 100 pixels is not exact, but close enough for our purpose here
            offset = 100 # offset for dst points
            # Grab the image shape
            img_size = (gray.shape[1], gray.shape[0])
    
            # For source points I'm grabbing the outer four detected corners
            src = np.float32([corners[0], corners[self.nx-1], corners[-1], corners[-self.nx]])
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result 
            # again, not exact, but close enough for our purposes
            dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                         [img_size[0]-offset, img_size[1]-offset], 
                                         [offset, img_size[1]-offset]])
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # Warp the image using OpenCV warpPerspective()
            warped = cv2.warpPerspective(undist, M, img_size)
        else:
            print ("Can not warp")
            warped = np.copy(img) 
            M = None
    
        # Return the resulting image and matrix
        return ret, warped, M
    

        
if __name__ == '__main__':
    c = CameraCalibration()
    #c.camera_calibration();
    ksize = 3
    
    img = cv2.imread("signs_vehicles_xygrad.png")  
    rgb = img[:,:,::-1]
    
    

    s_hls = c.hls_select(img, thresh=(170, 255))
    img2 = c.abs_sobel_thresh(img,thresh_min=20, thresh_max=100)
    img3 = c.mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30,100))
    img4 = c.dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    
    img5 = c.pipeline(rgb)
    
    c.show(rgb, s_hls, "Hls", img2, "Sobel abs", img3, "Sobel magn", \
           img4, "Sobel dir", img5, "Abs & HLS")
    
    print("done")
    