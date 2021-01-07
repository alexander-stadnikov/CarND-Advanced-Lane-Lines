from dataclasses import dataclass

import numpy as np
import cv2 as cv


@dataclass
class SlidingWindowsConfig:
    """ Properties for the Sliding Windows algorithm. """
    number_of_windows: int
    window_margin: int
    min_pixels_for_detect: int


class Detector:
    """ Finds lane lines, their curvature, CCP and overlays if needed. """
    
    def __init__(self, sliding_windows_config: SlidingWindowsConfig):
        self.cfg = sliding_windows_config

    def _find_bottom_peaks(self, img):        
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        return np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint        

    def _find_lane_pixels(self, img):
        out_img = np.dstack((img, img, img))
        
        window_height = np.int(img.shape[0] // self.cfg.number_of_windows)
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current, rightx_current = self._find_bottom_peaks(img)

        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.cfg.number_of_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - self.cfg.window_margin
            win_xleft_high = leftx_current + self.cfg.window_margin
            win_xright_low = rightx_current - self.cfg.window_margin
            win_xright_high = rightx_current + self.cfg.window_margin
            
            cv.rectangle(out_img,(win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv.rectangle(out_img,(win_xright_low,win_y_low),
                (win_xright_high,win_y_high),(0,255,0), 2) 
            
            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_y = (win_y_low <= nonzeroy) & (nonzeroy < win_y_high)
            good_left_inds = (good_y & (win_xleft_low <= nonzerox)
                & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = (good_y & (win_xright_low <= nonzerox)
                & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > self.cfg.min_pixels_for_detect:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.cfg.min_pixels_for_detect:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img


    def fit_polynomial(self, img):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self._find_lane_pixels(img)

        ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')

        return out_img
