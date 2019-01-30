from scipy import misc
from scipy import interpolate
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import re
#==================================================================

class BrainSection():
    '''
    Note that this class does not depend on the nib module
    '''

    def __init__(self,\
            dir_path  = None,\
            slice_n   = 0,\
            array     = None,\
            transform = None):

        self.dir_path= dir_path
        self.slice_n = slice_n
        self.array   = array
        self.affine_transformation = transform

        self.bw    = None
        self.bw_threshold = 20
        self.normalizing_constant = 2440008.4924061
        #self.boundary_quality = cv2.CHAIN_APPROX_NONE
        self.boundary_quality  = cv2.CHAIN_APPROX_SIMPLE
        self.bw_method = cv2.THRESH_BINARY
        #self.bw_method = cv2.THRESH_OTSUS
        self.boundary_extractor = cv2.RETR_TREE
        self.boundary          = None
        self.boundary_pixels   = None
        self.n_boundary_points = None
        self.splines  = []

        self.linear_map = None
        self.shift = None

        #==================================
        #self.initialize_map()
        #self.print_mapping()
        #self.convert_array_to_gray()


#==================================================================
    def initialize_map(self):

        #coronal_map_transpose
        self.affine_transformation[[0,2],[2,0]] =\
                self.affine_transformation[[2,0],[0,2]]

        '''
        No need to exchange the components of the shift vector
        '''
        #self.affine_transformation[[0,2],[3,3]] =\
                #self.affine_transformation[[2,0],[3,3]]
        #End coronal_map_transpose

        self.linear_map = self.affine_transformation[:3, :3]
        self.shift = self.affine_transformation[:3, -1].reshape(3,1)

        x = np.array([0., self.slice_n, 0., 1.])
        out = self.affine_transformation.dot(x)

        '''
        Component 1 is equivalent to the z coordinate 
        of the current model
        '''
        self.slice_z_position = out[1]

#==================================================================
    def print_mapping(self):

        print('Linear map')
        print (self.linear_map)
        print('Shift')
        print (self.shift)
        print('The coronal section {:d} is located at: {:0.3f} mm'.\
                format(self.slice_n, self.slice_z_position))

#==================================================================
    def convert_array_to_gray(self):
        if 255 < np.max(self.array):
            self.array *= 255 / np.max(self.array)
        if self.array.dtype != np.uint8:
            self.array = self.array.astype(np.uint8)

#==================================================================
    def plot_bw(self): 

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.imshow(self.bw, cmap='gray', origin='lower')

        base = 'bw.pdf'
        fname = os.path.join(self.dir_path, base)
        txt = 'Slice {:d} with z = {:0.3f} mm wrt bregma'.\
                format(self.slice_n, self.slice_z_position)
        fig.suptitle(txt, fontsize=18)

        plt.axis('off')
        fig.savefig(fname, dpi=300)

        plt.close('all')

#==================================================================
    def plot_grayscale(self): 

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.imshow(self.array, cmap='gray', origin='lower')

        base = 'gray.pdf'
        fname = os.path.join(self.dir_path, base)
        fig.savefig(fname, dpi=300)

        plt.close('all')


#==================================================================
    def store_boundary_pixels(self):

        fname = os.path.join(self.dir_path,\
                'boundary_pixel_data.txt')
        np.savetxt(fname, self.boundary_pixels, fmt = '%d')

#==================================================================
    def extract_boundary(self):
        threshold, self.bw = cv2.threshold(\
        self.array, self.bw_threshold, 255,\
        self.bw_method)
        

        _, contours, _ = cv2.findContours(\
                self.bw,\
                self.boundary_extractor,\
                self.boundary_quality)

        current_max_size = 0
        idx = 0
        n_contours = len(contours)

        if 1 < n_contours:
            print('We have {:d} contours to choose from'.\
                    format(n_contours))
            for k,c in enumerate(contours):
                current_contour_size = c.size
                if current_max_size < current_contour_size:
                    idx = k
                    current_max_size = current_contour_size

        self.boundary_pixels = contours[idx].reshape(-1,2).T


        '''
        Without dtype conversion data becomes corrupted.
        '''
        self.boundary = self.boundary_pixels.astype(np.float64)

        self.n_boundary_points = self.boundary.shape[1]

#==================================================================
    def map_to_bregma_center(self, points):

        y   = np.full(points[0].size, self.slice_n)
        return self.linear_map.dot([points[0], y, points[1]]) +\
                self.shift
        

#==================================================================
    def map_boundary_to_real(self):
        data = self.map_to_bregma_center(self.boundary)
        self.boundary[:] = data[[0,2]]

        if self.slice_z_position != data[1][0]:
            print('Possible discrepancy with z position')
            exit()

#==================================================================
    def sort_boundary(self, boundary):

        x = boundary[0]
        n = x.size
        indices = []
        indices.append(0)
        idx = 1

        while idx < n:
            current_x = x[idx]
            last_x    = x[indices[-1]] 

            if current_x == last_x:
                idx += 1
                continue

            if current_x < last_x:
                indices.pop()
                continue

            indices.append(idx)
            idx += 1

        return boundary[:,indices]




#==================================================================
    def separate_top_and_bottom_boundary(self):
        indices = np.argsort(self.boundary[0])
        min_idx = indices[0]
        max_idx = indices[-1]

        if min_idx < max_idx:
            top_indices = np.arange(min_idx,max_idx+1)

            bottom_indices = np.hstack((\
                    np.arange(max_idx,self.n_boundary_points),\
                    np.arange(0,min_idx+1),\
                    ))

        else:
            top_indices = np.hstack((\
                    np.arange(min_idx,self.n_boundary_points),\
                    np.arange(0,max_idx+1),\
                    ))
            bottom_indices = np.arange(max_idx,min_idx+1)

        bottom_indices = bottom_indices[::-1]
        self.top_boundary = self.boundary[:,top_indices]
        self.bottom_boundary = self.boundary[:,bottom_indices]

#==================================================================
    def smooth_boundary(self):

        self.top_boundary    =\
                self.sort_boundary(self.top_boundary)
        self.bottom_boundary =\
                self.sort_boundary(self.bottom_boundary)

        #Matching boundary conditions
        self.bottom_boundary[:,[0,-1]] = self.top_boundary[:,[0,-1]]

#==================================================================
    def load_splines(self):

        self.splines = []
        splines      = []

        pattern = 'spline_(?P<spline_n>[0-9]+)'+\
                '_component_(?P<component_n>[0-9]).txt'
        spline_regex = re.compile(pattern)

        for f_name in os.listdir(self.dir_path): 
            
            '''
            Extract spline number and component number
            '''
            obj = spline_regex.match(f_name)

            if obj is None:
                continue

            spline_n    = int(obj.group('spline_n'))
            component_n = int(obj.group('component_n'))

            while len(splines) <= spline_n:
                splines.append([])
            while len(splines[spline_n]) <= component_n:
                splines[spline_n].append([])

            fname = os.path.join(self.dir_path, f_name)
            component = np.loadtxt(fname) 
            splines[spline_n][component_n] = component

        for s in splines: 
            self.splines.append(tuple(s))

        print('The splines were loaded')

##==================================================================
    def save_splines(self):

        for s_idx, s in enumerate(self.splines):

            spline_base = 'spline_' + str(s_idx)

            for c_index, c in enumerate(s):
                file_base =\
                        spline_base   +\
                        '_component_' +\
                        str(c_index)  + '.txt'

                fname = os.path.join(self.dir_path, file_base)

                if type(c) != np.ndarray:
                    obj = np.array([c])
                else:
                    '''
                    No copy is generated
                    '''
                    obj = c

                np.savetxt(fname, obj)

#==================================================================
    def create_boundary_and_splines(self):

        self.extract_boundary()
        obj.store_boundary_pixels()
        self.map_boundary_to_real()
        self.separate_top_and_bottom_boundary()
        self.smooth_boundary()


        self.splines.append(\
                interpolate.splrep(\
                self.bottom_boundary[0],\
                self.bottom_boundary[1],\
                s=0))

        self.splines.append(\
                interpolate.splrep(\
                self.top_boundary[0],\
                self.top_boundary[1],\
                s=0))

        self.save_splines()

#==================================================================
    def test_splines(self):
        fig = plt.figure()
        ax  = fig.add_subplot(111)

        x_size = 25

        x = np.linspace(\
                self.top_boundary[0,0],\
                self.top_boundary[0,-1],\
                x_size)

        for s in self.splines:
            y = interpolate.splev(x, s)
            ax.plot(x, y, linewidth=2)

        base = 'spline.pdf'
        fname = os.path.join(self.dir_path, base)
        fig.savefig(fname, dpi=300)

        plt.close('all')

#==================================================================
    def generate_mesh(self):

        x_size = 35
        y_size = 30

        x_min  = self.splines[0][0][0]
        x_max  = self.splines[0][0][-1]

        x = np.linspace(\
                x_min,\
                x_max,\
                x_size)
        y = np.linspace(0,1,y_size)

        y_top = interpolate.splev(x, self.splines[1])
        y_bottom = interpolate.splev(x, self.splines[0])

        print('The mesh was generated')

        X,Y =  np.meshgrid(x,y)

        for k in range(y_top.size):
            yt = y_top[k]
            yb = y_bottom[k]
            y = np.linspace(yb,yt,y_size)
            Y[:,k] = y

        return (X,Y)


#==================================================================
    def test_spline_slice(self, fun=None):
        fig = plt.figure()
        ax  = fig.add_subplot(111)

        x_size = 25
        y_size = 20

        x_size = 35
        y_size = 30
        #indices = np.argsort(self.splines[0][0])
        x_min  = self.splines[0][0][0]
        x_max  = self.splines[0][0][-1]

        x = np.linspace(\
                x_min,\
                x_max,\
                x_size)
        y = np.linspace(0,1,y_size)

        y_top = interpolate.splev(x, self.splines[1])
        y_bottom = interpolate.splev(x, self.splines[0])

        X,Y = np.meshgrid(x,y)

        for k in range(y_top.size):
            yt = y_top[k]
            yb = y_bottom[k]
            y = np.linspace(yb,yt,y_size)
            Y[:,k] = y

        ax.plot(x, y_top,\
                'b-', linewidth=2)
        ax.plot(x, y_bottom,\
                'r-', linewidth=2)

        #Z = X**2 + Y**2
        Z = fun(X,Y)

        ax.pcolor(X,Y,Z)

        base = 'grid.pdf'
        fname = os.path.join(self.dir_path, base)
        fig.savefig(fname, dpi=300)

        plt.close('all')

#==================================================================
    def plot_boundary(self):

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        #c = np.vstack((c,c[0]))
        ax.plot(self.boundary[0], self.boundary[1],\
                'k+', linewidth=2)

        base = 'bdry.pdf'
        fname = os.path.join(self.dir_path, base)
        fig.savefig(fname, dpi=300)

        plt.close('all')

#==========================================================
