import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#sys.path.insert(0, '.')
class SliceProperties():

#==================================================================

    def __init__(self):
        self.experimental_z_n = None
        self.model_z_n = None
        self.model_z_mm = None
        self.height_pixel= None
        self.width_pixel = None
        self.height_mm= None
        self.width_mm = None
        self.channel = None
        self.xy_pixel_center = None
        self.experimental_center_in_model_mm = None
#==================================================================

    def map_from_model_xy_to_experimental_matrix(self, x, y):
        '''
        Change to coordinates wrt the experimental center
        '''

        if self.channel is None:
            return np.zeros(x.shape)

        xc = x.copy()
        yc = y.copy()
        xc -= self.experimental_center_in_model_mm[0]
        yc -= self.experimental_center_in_model_mm[1]
        xc *=  self.width_pixel/self.width_mm
        yc *= -self.height_pixel/self.height_mm
        xc += self.xy_pixel_center[0]
        yc += self.xy_pixel_center[1]
        xc = xc.astype(int)
        yc = yc.astype(int)

        '''
        Bring inside the domain
        '''
        xc[xc < 0] = 0
        yc[yc < 0] = 0
        xc[xc > self.width_pixel-1] =\
                self.width_pixel-1
        yc[yc > self.height_pixel-1] =\
                self.height_pixel-1

        indices = yc * self.width_pixel + xc

        return self.channel.flat[indices].astype(np.float64)


