import sys
import nibabel as nib
import numpy as np
from scipy.interpolate import CubicSpline as cspline
from scipy.stats import mode as get_mode
import os
import matplotlib.pyplot as plt
from matplotlib import cm as color_map
from matplotlib.colors import LinearSegmentedColormap as lsc
from mpl_toolkits.mplot3d import Axes3D
import re
import cv2

#sys.path.insert(0, '.')
from timing_module import TimeIt
from brain_slice_module import BrainSection
from slice_properties_module import SliceProperties

#==================================================================

class ImageManipulation():

    def __init__(self):

        self.nib_file   = None

        self.nib_is_loaded = False

        self.affine_transformation = None

        '''
        XY coordinates of the experimental center with
        respect to the model coordinate system.
        '''
        self.xy_experimental_center_in_model_mm =\
                np.array([2,0]) 
        
        '''
        From top of the brain (bregma) to site of injection
        '''
        self.depth_of_injection_in_mm = 3.0

        '''
        Location of coronal section at the site of injection
        using experimental coordinates in mm
        '''
        self.site_of_injection_in_experimental_z_mm = 0

        self.safe_range = (200, 1000)

        self.main_dir = os.getcwd()

        self.experimental_data_dir = os.path.join(\
                self.main_dir, 'experimental_data')

        self.postprocessing_dir = os.path.join(\
                self.main_dir, 'postprocessing')

        self.brain_mesh_local_dir = 'brain_model'

        self.brain_mesh_dir = os.path.join(\
                self.main_dir, self.brain_mesh_local_dir)

        self.brain_slices_txt_local_dir = 'slices_txt'

        self.brain_slices_txt_dir = os.path.join(\
                self.main_dir, self.brain_slices_txt_local_dir)

        self.brain_slices_local_dir = 'slices'

        self.brain_slices_dir = os.path.join(\
                self.main_dir, self.brain_slices_local_dir)

        self.map_experimental_z_n_to_mm =\
                lambda z_n:\
                {0:-4., 1:-2., 2:0., 3:0.6}.get(z_n, None)

        self.map_experimental_z_mm_to_model_z_mm =\
                lambda z_mm: -1.5 - z_mm

        self.map_model_z_mm_to_experimental_z_mm =\
                lambda z_mm: -1.5 - z_mm

        '''
        XYZ coordinates of the experimental center with
        respect to the model coordinate system.
        '''
        self.site_of_injection_in_model_mm = None

        self.slice_n_at_injection_site = None

        self.slice_properties          = []
        self.list_of_experimental_z_mm = []
        self.sorted_slice_properties   = []
        self.sorted_experimental_z_mm  = None
        self.n_of_slice_objects        = 0
#==================================================================M

        self.load_nib_affine_transformation()
        self.compute_injection_site_properties()

#==================================================================
    def compute_injection_site_properties(self):

        b    = self.site_of_injection_in_experimental_z_mm
        z_mm = self.map_experimental_z_mm_to_model_z_mm(b)
        z_n  = self.map_model_z_mm_to_z_n(z_mm)
        self.slice_n_at_injection_site = z_n

        a = self.xy_experimental_center_in_model_mm

        self.site_of_injection_in_model_mm =\
                np.concatenate((a,[z_mm])) 


#==================================================================
    def map_z_mm_at_experimental_center_to_pixel_xy(self, z_mm):

        c_mm = np.array([\
                self.xy_experimental_center_in_model_mm[0],\
                z_mm,\
                self.xy_experimental_center_in_model_mm[1],\
                1,\
                ])
        voxel = np.linalg.solve(\
                self.affine_transformation,\
                c_mm)[:3]

        pixel = voxel[[0,2]].astype(int)

        return pixel


#==================================================================
    def up_one_directory(self, dir_path=os.getcwd() ):
        regex = re.compile(r'(?P<stem>.*)/\w+')
        obj = regex.search(dir_path)
        if obj is not None:
            stem = obj.group('stem')
            return stem

#==================================================================
    def load_nib_file(self):

        if self.nib_is_loaded:
            return

        brain_mesh_bname = 'brain_model.nii'
        brain_mesh_fname = os.path.join(\
                self.brain_mesh_dir, brain_mesh_bname)

        if not os.path.exists(brain_mesh_fname):
            print('File {:s} does not exists'.\
                    format(brain_mesh_fname))

        self.nib_file = nib.load(brain_mesh_fname)

        self.nib_is_loaded = True

#==================================================================
    def generate_text_files_of_grayscales(self):

        with TimeIt('Load file'):
            if not self.nib_is_loaded:
                self.load_nib_file()
            shape = self.nib_file.shape
            n_coronal_sections  = shape[1]
            n_sagittal_sections = shape[0]

        with TimeIt('Convert nib to numpy'): 
            data = self.nib_file.get_fdata()

        with TimeIt('Convert full array to gray data'):
            data = img.get_fdata()
            data *= 255 / data.max()
            data = data.astype('uint8')

        with TimeIt('Save the entire matrix'): 
            for k in range(n_coronal_sections):
                local_fname = 'slice_' + str(k) + '_matrix.txt'
                slice_data_fname = os.path.join(\
                        self.brain_slices_txt_dir, local_fname)
                m = data[:,k,:].T
                np.savetxt(slice_data_fname, m, fmt = '%d')
                print('Just saved slice # {:d}'.format(k))
                print('{:0.3f} %'.\
                        format(k/n_coronal_sections*100))
                print('------------')

        print('We have {:d} coronal sections'.\
                format(n_coronal_sections))

#==================================================================
    def load_nib_affine_transformation(self):

        brain_transformation_bname = 'affine.txt'
        brain_transformation_fname = os.path.join(\
                self.brain_mesh_dir, brain_transformation_bname)

        if not os.path.exists(brain_transformation_fname):

            self.load_nib_file()

            self.affine_transformation = self.nib_file.affine
            np.savetxt(brain_transformation_fname,\
                    self.affine_transformation)

        else:

            self.affine_transformation = np.loadtxt(\
                    brain_transformation_fname)


#==================================================================
    def create_boundary_data(self):

        self.load_nib_file()

        number_regex = re.compile(r'[0-9]+')

        for (dir_path, dir_names, file_names) in\
                os.walk(self.brain_slices_dir):

            print('Working in {:s}'.format(dir_path))
            basename = os.path.basename(dir_path)
            obj = number_regex.search(basename)

            if obj is None:
                continue

            idx = int(obj.group(0))
            if (idx < self.safe_range[0]) or\
                    (self.safe_range[1] < idx):
                continue

            fname = basename + '.txt'
            source_path =\
                    os.path.join(\
                    self.brain_slices_txt_dir, fname)

            '''
            We are loading the gray image data from a text 
            file to be read with numpy.
            '''

            if not os.path.exists(source_path):
                print('File {:s} does not exists'.\
                        format(source_path))

            gray = np.loadtxt(source_path, dtype='uint8')

            obj = BrainSection(\
                    dir_path,\
                    idx,\
                    gray,\
                    self.affine_transformation)

            obj.extract_boundary()
            obj.store_boundary_pixels()

#==================================================================
    def get_brain_slice_from_experimental_z_mm(self, z_mm):

        model_z_mm = self.map_experimental_z_mm_to_model_z_mm(z_mm)
        return self.get_brain_slice_from_model_z_mm(model_z_mm)

#==================================================================
    def get_brain_slice_from_model_z_mm(self, z_mm):

        z_n = self.map_model_z_mm_to_z_n(z_mm)
        return self.get_brain_slice_from_model_z_n(z_n)

#==================================================================
    def get_brain_slice_from_model_z_n(self, slice_idx):

        slice_data_dir = 'slice_' + str(slice_idx) + '_matrix'

        local_slice_dir =\
                os.path.join(\
                self.brain_slices_dir, slice_data_dir)

        brain_slice =\
                BrainSection(\
                local_slice_dir,\
                slice_idx)

        brain_slice.load_splines()

        return brain_slice

#==================================================================
    def create_boundary_and_spline_data(self):

        self.load_nib_file()

        number_regex = re.compile(r'[0-9]+')

        for (dir_path, dir_names, file_names) in\
                os.walk(self.brain_slices_dir):

            print('Working in {:s}'.format(dir_path))
            basename = os.path.basename(dir_path)
            obj = number_regex.search(basename)

            if obj is None:
                continue

            idx = int(obj.group(0))
            if (idx < self.safe_range[0]) or\
                    (self.safe_range[1] < idx):
                continue

            fname = basename + '.txt'
            source_path = os.path.join(\
                    self.brain_slices_txt_dir, fname)

            '''
            We are loading the gray image data from a text 
            file to be read with numpy.
            '''

            if not os.path.exists(source_path):
                print('File {:s} does not exists'.\
                        format(source_path))

            gray = np.loadtxt(source_path, dtype='uint8')
            obj = BrainSection(dir_path,\
                    idx,\
                    gray,\
                    self.affine_transformation)

            obj.create_boundary_and_splines()

#==================================================================
    def map_model_z_mm_to_z_n(self, z_mm):

        v = np.array([0,z_mm,0,1])
        backward = np.linalg.solve(self.affine_transformation, v)
        return np.round(backward[1]).astype(int)

#==================================================================
    def map_model_z_n_to_mm(self, z_n):

        v = np.array([0,z_n,0,1])
        forward = self.affine_transformation.dot(v)
        return forward[1]

#==================================================================
    def map_experimental_z_n_to_model_z_n(self, z_n):

        v = self.map_experimental_z_n_to_mm(z_n)
        v = self.map_experimental_z_mm_to_model_z_mm(v)
        v = self.map_model_z_mm_to_z_n(v)
        return np.round(v).astype(int)

#==================================================================
    def remove_white_borders(self, img, dir_path):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #min_value = gray.min()
        #max_value = gray.max()
        #print(min_value, max_value)
        #print(gray.shape)
        bw     = gray < 170
        coords = cv2.findNonZero(bw.astype(np.uint8))
        #print(coords.shape)
        x,y,w,h = cv2.boundingRect(coords)
        cropped_img  =  img[y:y+h,x:x+w]


        cropped_fname = 'cropped.jpeg'
        cropped_fname = os.path.join(\
                dir_path,\
                cropped_fname)

        cv2.imwrite(cropped_fname, cropped_img)

        return cropped_img

#==================================================================
    def interpolate_intensity(self, x, y, z,\
            using_experimental_coordinates_for_z = True):

        if not using_experimental_coordinates_for_z:
            z = self.map_model_z_mm_to_experimental_z_mm(z)

        sorted_z_indices = np.argsort(z)

        sorted_z = z[sorted_z_indices]
        sorted_x = x[sorted_z_indices]
        sorted_y = y[sorted_z_indices]
        u = np.zeros(z.size)

        n_elements  = z.size
        left_bound  = 0
        right_bound = 0

        for slice_idx in range(1,self.n_of_slice_objects):

            slice_z_mm =\
                    self.sorted_experimental_z_mm[slice_idx]
            print('Slice z mm: {:0.3f}'.format(slice_z_mm))

            interval_is_empty = True

            while (right_bound < n_elements) and\
                    (sorted_z[right_bound] <= slice_z_mm):
                right_bound      += 1
                interval_is_empty = False

            if interval_is_empty:
                continue 

            x_interval = sorted_x[left_bound:right_bound]
            y_interval = sorted_y[left_bound:right_bound]
            z_interval = sorted_z[left_bound:right_bound]

            obj_0 = self.sorted_slice_properties[slice_idx-1]
            obj_1 = self.sorted_slice_properties[slice_idx]

            z_0   = obj_0.experimental_z_mm
            z_1   = obj_1.experimental_z_mm
            delta_z = z_1 - z_0

            v_0 = obj_0.\
                    map_from_model_xy_to_experimental_matrix\
                    (x_interval,y_interval)

            v_1 = obj_1.\
                    map_from_model_xy_to_experimental_matrix\
                    (x_interval,y_interval)


            slope = (v_1 - v_0)/delta_z
            v =  (z_interval - z_0)* slope + v_0

            u[sorted_z_indices[left_bound:right_bound]] = v

            left_bound = right_bound

            if right_bound == n_elements:
                break

        return u


#==================================================================
    def plot_interpolated_slices(self):

        plot_in_3d = True
        #plot_in_3d = False
        data_is_in_mm = False

        self.slice_interpolation_setup()
        slices_to_plot = [self.slice_n_at_injection_site]
        slices_to_plot = np.linspace(400,765,10).astype(int)

        if data_is_in_mm: 
            slices_to_plot =\
                    [self.map_model_z_mm_to_z_n(z_mm)\
                    for z_mm in slices_to_plot]

        fig = plt.figure()
        if plot_in_3d:
            ax = fig.add_subplot(111, projection='3d')
            min_z = +np.inf
            max_z = -np.inf
        else:
            ax = fig.add_subplot(111)
        colors = [(0,0,128/255), (1,1,0), (1,1,1)]
        c_map_name = 'custom'
        c_map = lsc.from_list(c_map_name, colors, N = 100)

        for slice_idx in slices_to_plot:
            print('++++++++++++++++++++++++++++++++++')
            print('Plotting slice: {:d}'.format(slice_idx))

            slice_z_mm =\
                    self.map_model_z_n_to_mm(slice_idx)
            slice_z_mm_exp =\
                    self.map_model_z_mm_to_experimental_z_mm(\
                    slice_z_mm)

            print('Slice z_mm (model): {:0.3f}'.format(slice_z_mm))
            print('Slice z_mm (expr.): {:0.3f}'.\
                    format(slice_z_mm_exp))


            brain_slice = self.get_brain_slice_from_model_z_n(slice_idx)
            X,Y = brain_slice.generate_mesh_from_splines()

            U   = np.full(X.shape, slice_z_mm_exp)

            if plot_in_3d: 
                min_z = np.min((min_z,slice_z_mm_exp))
                max_z = np.max((max_z,slice_z_mm_exp))

            '''
            Note that the (x,y) values are in model coordinates, while the z
            values are in experimental coordinates.
            '''
            x = X.ravel()
            y = Y.ravel()
            z = U.ravel()

            u = self.interpolate_intensity(x, y, z)
            Z = u.reshape(X.shape)

            if plot_in_3d:
                c_map = color_map.jet(Z/255)
                ax.plot_surface(\
                        X,Y,U,\
                        facecolors = c_map,\
                        rstride = 1,\
                        cstride = 1,\
                        alpha = 0.3,\
                        linewidth = 0.05,\
                        #linewidth = 0.0,\
                        )
            else:
                c = ax.pcolor(X,Y,Z,\
                        cmap = c_map, vmin=0, vmax=255)
                fig.colorbar(c, ax=ax)
                txt = 'slice_' + str(slice_idx) + '.pdf'
                fig.savefig(txt, dpi = 300)


        if plot_in_3d:
            main_c_map =\
                    color_map.ScalarMappable(\
                    cmap = color_map.jet)
            a,b,c = self.site_of_injection_in_model_mm
            ax.scatter(a,b,c, color='k', marker='o', s = 10)
            z_bottom = min_z - 0.1
            z_top    = max_z + 0.1
            ax.plot([a,a],[b,b],\
                    [z_bottom,z_top],\
                    linewidth=2,color='k')

            ax.yaxis.set_ticks([-3,0,3])
            ax.set_zlim(z_bottom, z_top)
            ax.set_xlabel('Sagittal')
            ax.set_ylabel('Transverse')
            ax.set_zlabel('Coronal')

            fname = 'brain_mesh.pdf'
            fname = os.path.join(\
                    self.postprocessing_dir,\
                    fname)
            fig.savefig(fname, dpi = 300)

#==================================================================
    def plot_sphere(self):

        n=40

        self.slice_interpolation_setup()

        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')

        theta = np.linspace(0,2*np.pi,n)
        phi   = np.linspace(0,1*np.pi,n)

        theta = np.linspace(-np.pi/2,1*np.pi,n)
        phi   = np.linspace(0,1*np.pi,n)

        T,P   = np.meshgrid(theta, phi)
        a,b,c = self.site_of_injection_in_model_mm

        dr = 0.25
        rho_vector = np.arange(dr,2+dr,dr)
        rho_vector = [0.5, 1, 2]
        alpha_vector = [0.3, 0.6, 0.3]

        for k, rho in enumerate(rho_vector):

            X = rho * np.cos(T) * np.sin(P) + a
            Y = rho * np.sin(T) * np.sin(P) + b
            Z = rho * np.cos(P)             + c

            x = X.ravel()
            y = Y.ravel()
            z = Z.ravel()

            z = self.map_model_z_mm_to_experimental_z_mm(z)

            u = self.interpolate_intensity(x,y,z)
            U = u.reshape(X.shape)



            #ax.plot_surface(X,Y,Z)

            c_map = color_map.jet(U/255)
            ax.plot_surface(\
                    X,Y,Z,\
                    facecolors = c_map,\
                    rstride = 1,\
                    cstride = 1,\
                    alpha = alpha_vector[k],\
                    #linewidth = 0.05,\
                    linewidth = 0.0,\
                    )

        ax.set_xlabel('Sagittal')
        ax.set_ylabel('Transverse')
        ax.set_zlabel('Coronal') 

        ax.xaxis.set_ticks([0,2,4])
        ax.yaxis.set_ticks([0,2])
        ax.zaxis.set_ticks([-1.5, 0])
        ax.view_init(elev=15, azim=-150)

        fname = 'sphere.pdf'
        fname = os.path.join(\
                self.postprocessing_dir,\
                fname)
        fig.savefig(fname, dpi = 300)

#==================================================================
    def get_radial_plot(self):


        self.slice_interpolation_setup()

        n = 40
        theta = np.linspace(0,2*np.pi,n)
        phi   = np.linspace(0,1*np.pi,n)

        dr = 0.01
        rho_max = 4
        rho   = np.arange(0,rho_max,dr)

        T,P = np.meshgrid(theta, phi)
        a,b,c = self.site_of_injection_in_model_mm

        X = np.cos(T) * np.sin(P)
        Y = np.sin(T) * np.sin(P)
        Z = np.cos(P)

        s = 0

        cumulative_vec = np.zeros((rho.size,2))

        for k,r in enumerate(rho):

            x = r * X.ravel() + a
            y = r * Y.ravel() + b
            z = r * Z.ravel() + c

            z = self.map_model_z_mm_to_experimental_z_mm(z)
            u = self.interpolate_intensity(x,y,z)

            mode = get_mode(u).mode[0]
            mean = np.mean(u)
            mx   = u.max()
            threshold = np.max((mode,mean))
            print('Mean = {:0.1f}, Mode = {:0.1f}, Max = {:0.1f}'.\
                    format(mean, mode, mx))
            #u *= threshold < u
            u /= 255
            s += u.sum()
            cumulative_vec[k] = [r, s]


        x = cumulative_vec[:,0]
        y = cumulative_vec[:,1]

        stride = 4
        spline = cspline(\
                x[::stride],\
                y[::stride],\
                )
        spline_prime = spline.derivative()

        xp = np.linspace(0,rho_max,100)
        yp = spline_prime(xp)

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(x,y,'b-')
        fig.suptitle('Cumulative intensity')
        ax.set_xlabel('Distance from injection (mm)')
        txt  = r'$C(r)='
        txt += r'\int_{0}^{r}'
        txt += r'\int_{0}^{\pi}'
        txt += r'\int_{0}^{2\pi}'
        txt += r'I(\rho,\phi,\theta)'
        txt += r'\, d\theta \, d\phi \, d\rho$'
        ax.set_ylabel(txt)
        ax.yaxis.set_ticks(np.linspace(0,y.max(),4))
        ax.set_yscale('log')
        #fig.canvas.draw()
        #yticklabels = ax.get_yticklabels()
        #yticklabels = [label.get_text() for label in yticklabels]
        #yticklabels = ['{:.0e}'.format(float(label)) for label in yticklabels]
        #ax.set_yticklabels(yticklabels)
        fname = 'integral_radial.pdf'
        fname = os.path.join(\
                self.postprocessing_dir,\
                fname)
        fig.savefig(fname, dpi = 300)

        fig.clf()
        ax = fig.add_subplot(111)

        ax.plot(xp,yp,'b-')
        fig.suptitle('Radial intensity')
        ax.set_xlabel('Distance from injection (mm)')
        fig.canvas.draw()
        yticklabels = ax.get_yticklabels()
        yticklabels = [label.get_text() for label in yticklabels]
        yticklabels = [float(label) for label in yticklabels]
        yticklabels = ['{:.0e}'.format(label) for label in yticklabels]
        ax.set_yticklabels(yticklabels)
        txt  = r'$dC/d\rho$'
        ax.set_ylabel(txt)
        
        fname = 'derivative_radial.pdf'
        fname = os.path.join(\
                self.postprocessing_dir,\
                fname)
        fig.savefig(fname, dpi = 300)

#==================================================================
    def slice_interpolation_setup(self):

        self.load_experimental_data_essentials()

        self.add_zero_slices_from_safe_range()

        self.list_of_experimental_z_mm =\
                np.array(self.list_of_experimental_z_mm)

        self.sorted_indices_experimental_z_mm =\
                np.argsort(self.list_of_experimental_z_mm)

        self.sorted_experimental_z_mm =\
                self.list_of_experimental_z_mm[\
                self.sorted_indices_experimental_z_mm]

        self.n_of_slice_objects =\
                len(self.list_of_experimental_z_mm)

        self.sorted_slice_properties =\
                [self.slice_properties[idx]\
                for idx in self.sorted_indices_experimental_z_mm]

        print('The sorted indices are:')
        print(self.sorted_indices_experimental_z_mm)

        print('Experimental z_mm in experimental coordinates:')
        print(self.sorted_experimental_z_mm)

        print('Experimental z_mm in model coordinates:')
        print(self.map_experimental_z_mm_to_model_z_mm(\
                self.sorted_experimental_z_mm))

#==================================================================
    def add_zero_slices_from_safe_range(self):

        for v in self.safe_range:

            model_z_mm = self.map_model_z_n_to_mm(v)
            SP = SliceProperties()
            SP.model_z_mm = model_z_mm

            experimental_z_mm =\
                    self.map_model_z_mm_to_experimental_z_mm(\
                    model_z_mm)

            SP.experimental_z_mm = experimental_z_mm

            self.slice_properties.append(SP)
            self.list_of_experimental_z_mm.append(\
                    experimental_z_mm)

        infinity_range = (-1000, 1000)
        for v in infinity_range:

            model_z_mm = self.map_model_z_n_to_mm(v)
            SP = SliceProperties()
            SP.model_z_mm = model_z_mm

            experimental_z_mm =\
                    self.map_model_z_mm_to_experimental_z_mm(\
                    model_z_mm)

            SP.experimental_z_mm = experimental_z_mm

            self.slice_properties.append(SP)
            self.list_of_experimental_z_mm.append(\
                    experimental_z_mm)


#==================================================================
    def load_experimental_data_essentials(self):

        z_slice_n_regex = re.compile(r'm_(?P<z_n>[0-9]+)')
        for (dir_path, dir_names, file_names) in\
                os.walk(self.experimental_data_dir):

            '''
            Name extraction
            '''
            print('Working in {:s}'.format(dir_path))
            experimental_basename =\
                    os.path.basename(dir_path)
            obj = z_slice_n_regex.search(experimental_basename)

            if obj is None:
                continue

            SP = SliceProperties()
            SP.xy_experimental_center_in_model_mm =\
                    self.xy_experimental_center_in_model_mm

            experimental_z_n = int(obj.group('z_n'))
            SP.experimental_z_n = experimental_z_n
            print('The expr. z_n  = {:d}'.\
                    format(experimental_z_n))

            experimental_z_mm =\
                    self.map_experimental_z_n_to_mm(\
                    experimental_z_n)
            SP.experimental_z_mm = experimental_z_mm
            print('The expr. z_mm  = {:0.3f}'.\
                    format(experimental_z_mm))

            model_z_n =\
                    self.map_experimental_z_n_to_model_z_n(\
                    experimental_z_n)
            print('The model z_n  = {:d}'.format(model_z_n))
            SP.model_z_n = model_z_n

            model_z_mm =\
                    self.map_model_z_n_to_mm(\
                    model_z_n)
            print('The model z_mm = {:0.3f}'.format(model_z_mm))
            SP.model_z_mm = model_z_mm

            '''
            Load resized image dimensions in pixels
            '''
            height_width_in_pixels_fname =\
                    'height_width_in_pixels.txt'
            height_width_in_pixels_fname = os.path.join(\
                    dir_path,\
                    height_width_in_pixels_fname)

            h,w = np.loadtxt(\
                    height_width_in_pixels_fname,\
                    dtype=int)
            SP.height_pixel= h
            SP.width_pixel = w

            '''
            Load resized image dimensions in mm
            '''
            height_width_in_mm_fname = 'height_width_in_mm.txt'
            height_width_in_mm_fname = os.path.join(\
                    dir_path,\
                    height_width_in_mm_fname)

            h,w = np.loadtxt(height_width_in_mm_fname)
            SP.height_mm= h
            SP.width_mm = w

            '''
            Load grayscale matrix of red channel
            '''
            channel_fname = 'channel.txt'
            channel_fname = os.path.join(\
                    dir_path,\
                    channel_fname)
            SP.channel = \
                    np.loadtxt(channel_fname, dtype=np.uint8)

            '''
            Load center (X,Y) in pixel coordinates of
            experimental image.
            Zero is the upper left corner.
            '''
            center_fname = 'center.txt'
            center_fname = os.path.join(\
                    dir_path,\
                    center_fname)

            SP.xy_pixel_center = np.loadtxt(center_fname)

            self.slice_properties.append(SP)

            self.list_of_experimental_z_mm.append(\
                    SP.experimental_z_mm)

#==================================================================
    def generate_experimental_data_essentials(self):

        number_regex = re.compile(r'[0-9]+')
        z_slice_n_regex = re.compile(r'm_(?P<z_n>[0-9]+)')
        v = np.linalg.solve(\
                self.affine_transformation, np.array([0,0,0,1]))
        print('Model center in pixels')
        print(v)

        '''
        The coordinate system used in the experimental data
        considers the site of injection as the origin (located 1.5
        mm rostral to the bregma) and takes the rostral direction as
        positive.

        This maps send the coordinates of the experimental data in
        mm to coordinates in the model also in mm.  Note that in the
        model the caudal direction is the positive z-axis.
        '''

        for (dir_path, dir_names, file_names) in\
                os.walk(self.experimental_data_dir):

            '''
            Name extraction
            '''
            print('Working in {:s}'.format(dir_path))
            experimental_basename =\
                    os.path.basename(dir_path)
            obj = z_slice_n_regex.search(experimental_basename)

            if obj is None:
                continue

            experimental_z_n = int(obj.group('z_n'))
            print('Working with experimental slice {:d}'.\
                    format(experimental_z_n))


            '''
            Load experimental image based on z-axis parameters
            '''
            model_z_n =\
                    self.map_experimental_z_n_to_model_z_n(\
                    experimental_z_n)
            print('The model z_n = {:d}'.format(model_z_n))

            model_z_mm =\
                    self.map_model_z_n_to_mm(\
                    model_z_n)
            print('The model z_mm = {:0.3f}'.format(model_z_mm))

            experimental_fname =\
                    experimental_basename + '_original.jpeg'
            experimental_fname = os.path.join(\
                    dir_path,\
                    experimental_fname)

            if not os.path.exists(experimental_fname):
                print('Image {:s} does not exists'.format(fname))
                continue

            '''
            Load experimental image
            '''
            print('Reading experimental image')
            experimental_img = cv2.imread(experimental_fname) 
            
            '''
            Remove white borders
            '''
            experimental_img =\
                    self.remove_white_borders(\
                    experimental_img, dir_path)

            experimental_img_shape = experimental_img.shape[:2]

            print('Experimental pixel dimensions')
            print('H x W: {:d} x {:d}'.\
                    format(*experimental_img_shape))
            experimental_heigh_to_width_ratio =\
                    experimental_img_shape[0] /\
                    experimental_img_shape[1] 
            print('Experimental H:W')
            print(experimental_heigh_to_width_ratio)


            '''
            Loading Model boundary
            '''
            print('Loading model boundary {:d}'.\
                    format(model_z_n))

            slice_data_dir = 'slice_' + str(model_z_n) + '_matrix'

            local_slice_dir =\
                    os.path.join(\
                    self.brain_slices_dir, slice_data_dir)

            if not os.path.exists(local_slice_dir):
                print('Directory {:s} does not exists'.\
                        format(local_slice_dir))
                exit()

            boundary_fname = 'boundary_pixel_data.txt'

            boundary_fname =\
                    os.path.join(local_slice_dir, boundary_fname)

            if not os.path.exists(boundary_fname):
                print('Directory {:s} does not exists'.\
                        format(boundary_fname))
                exit()


            boundary = np.loadtxt(boundary_fname,\
                    dtype = int)

            '''
            Get bounding box of model data
            '''
            x,y,w,h = cv2.boundingRect(boundary.T)
            model_img_shape = (h, w)
            print('Model pixel dimensions')
            print('H x W: {:d} x {:d}'.\
                    format(*model_img_shape))

            x_indices = np.argsort(boundary[0])
            y_indices = np.argsort(boundary[1])

            pixel_at_x_min = boundary[:,x_indices[0]]
            pixel_at_x_max = boundary[:,x_indices[-1]]
            pixel_at_y_min = boundary[:,y_indices[0]]
            pixel_at_y_max = boundary[:,y_indices[-1]]

            print('Pixel at x min')
            print(pixel_at_x_min)
            print('Pixel at x max')
            print(pixel_at_x_max)
            print('Pixel at y min')
            print(pixel_at_y_min)
            print('Pixel at y max')
            print(pixel_at_y_max)

            '''
            Horizontal length
            '''
            left_pixel = np.array([\
                    pixel_at_x_min[0],\
                    model_z_n,\
                    pixel_at_x_min[1],\
                    1])
            left_mm = \
                    self.affine_transformation.dot(left_pixel)
            print('Model left mm')
            print(left_mm)

            right_pixel = np.array([\
                    pixel_at_x_max[0],\
                    model_z_n,\
                    pixel_at_x_max[1],\
                    1])
            right_mm = \
                    self.affine_transformation.dot(right_pixel)
            print('Model right mm')
            print(right_mm)

            model_width_mm = (right_mm - left_mm)[0]
            print('Model horizontal length mm')
            print(model_width_mm)

            '''
            Vertical length
            '''
            bottom_pixel = np.array([\
                    pixel_at_y_min[0],\
                    model_z_n,\
                    pixel_at_y_min[1],\
                    1])
            bottom_mm = \
                    self.affine_transformation.dot(bottom_pixel)
            print('Model bottom mm')
            print(bottom_mm)

            top_pixel = np.array([\
                    pixel_at_y_max[0],\
                    model_z_n,\
                    pixel_at_y_max[1],\
                    1])
            top_mm = \
                    self.affine_transformation.dot(top_pixel)
            print('Model top mm')
            print(top_mm)

            model_height_mm = (top_mm - bottom_mm)[2]
            print('Model vertical length mm')
            print(model_height_mm)

            model_height_width_in_mm = (\
                    model_height_mm,\
                    model_width_mm)

            '''
            Model aspect ratio
            '''
            model_heigh_to_width_ratio =\
                    model_img_shape[0] /\
                    model_img_shape[1] 
            print('Model aspect ratio H:W')
            print(model_heigh_to_width_ratio)


            '''
            Plot overlap
            '''
            overlap_img =\
                    cv2.resize(\
                    experimental_img,\
                    model_img_shape[::-1])

            y_img = h - (boundary[1]-boundary[1].min() + 1)
            x_img = boundary[0]-boundary[0].min()
            overlap_img[y_img, x_img, :] = 255

            '''
            Experimental theoretical center
            '''
            c = self.\
                    map_z_mm_at_experimental_center_to_pixel_xy(\
                    model_z_mm)

            y_img = h - (c[1] - boundary[1].min() + 1)
            x_img = c[0] - boundary[0].min()
            overlap_img[y_img, x_img, :] = 255

            overlap_fname = 'overlap.jpeg'
                    #experimental_basename + '_scaled.jpeg'
            overlap_fname = os.path.join(\
                    dir_path,\
                    overlap_fname)

            cv2.imwrite(overlap_fname, overlap_img)
            overlap_img = None



            '''
            Compute experimental image width 
            to match model aspect ratio
            '''
            new_experimental_width =\
                    experimental_img_shape[0] /\
                    model_heigh_to_width_ratio

            new_experimental_width =\
                    np.round(new_experimental_width).astype(int)

            experimental_img_shape = (\
                    experimental_img_shape[0],\
                    new_experimental_width,\
                    )

            print('New experimental pixel dimensions')
            print('H x W: {:d} x {:d}'.\
                    format(*experimental_img_shape))

            '''
            Store resized image shape 
            '''
            height_width_in_pixels_fname =\
                    'height_width_in_pixels.txt'
            height_width_in_pixels_fname = os.path.join(\
                    dir_path,\
                    height_width_in_pixels_fname)

            np.savetxt(height_width_in_pixels_fname,\
                    experimental_img_shape,\
                    fmt='%d')

            '''
            Store resized image dimensions in mm 
            '''
            height_width_in_mm_fname = 'height_width_in_mm.txt'
                    #experimental_basename + '_scaled.jpeg'
            height_width_in_mm_fname = os.path.join(\
                    dir_path,\
                    height_width_in_mm_fname)

            np.savetxt(height_width_in_mm_fname,\
                    model_height_width_in_mm,\
                    fmt='%f');

            '''
            Resize experimental image to match model aspect ratio
            '''
            experimental_img =\
                    cv2.resize(\
                    experimental_img,\
                    experimental_img_shape[::-1])
            '''
            Theoretical center 
            x = Midpoint of image plus 2mm to the right
            y = 3mm below surface
            '''
            experimental_height_widht_mm_to_pixel =\
                    np.array(experimental_img_shape) / \
                    np.array(model_height_width_in_mm)
            x  = experimental_img_shape[1] / 2
            x += experimental_height_widht_mm_to_pixel[1] * \
                    self.xy_experimental_center_in_model_mm[0]
            y  = experimental_height_widht_mm_to_pixel[0] * \
                    self.depth_of_injection_in_mm
            x  = np.round(x).astype(int)
            y  = np.round(y).astype(int)

            scaled_fname = 'scaled.jpeg'
            scaled_fname = os.path.join(\
                    dir_path,\
                    scaled_fname)

            cv2.imwrite(scaled_fname,\
                    experimental_img[:,:,:])

            '''
            Store red channel in jpeg and matrix format
            Recall that the output format is BGR
            '''
            channel = experimental_img[:,:,2]
            channel_fname = 'channel.jpeg'
            channel_fname = os.path.join(\
                    dir_path,\
                    channel_fname)

            cv2.imwrite(channel_fname,\
                    channel)
            channel_fname = 'channel.txt'
            channel_fname = os.path.join(\
                    dir_path,\
                    channel_fname)
            np.savetxt(channel_fname, channel, fmt='%d')



            '''
            Center extraction
            Row, column pixel location using matrix coordinates
            '''
            center_fname = 'center.txt'
            center_fname = os.path.join(\
                    dir_path,\
                    center_fname)

            if not os.path.exists(center_fname):
                print('File {:s} does not exists'.\
                        format(center_fname))
                print('=========================')
                continue
            experimental_center_pixel = np.loadtxt(center_fname)
            experimental_center_pixel =\
                    np.round(experimental_center_pixel).astype(int)
            x = experimental_center_pixel[0]
            y = experimental_center_pixel[1]
            print('User provided experimental center (Row,Col)')
            print(experimental_center_pixel)

            experimental_img_with_center =\
                    experimental_img.copy()
            cv2.circle(\
                    experimental_img_with_center,\
                    (x,y),\
                    10,\
                    (255,255,255),\
                    3,\
                    )

            scaled_with_center_fname = 'scaled_with_center.jpeg'
            scaled_with_center_fname = os.path.join(\
                    dir_path,\
                    scaled_with_center_fname)

            cv2.imwrite(scaled_with_center_fname,\
                    experimental_img_with_center[:,:,:])
            experimental_img_with_center = None





            
        


