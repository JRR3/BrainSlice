import sys
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import cv2

#sys.path.insert(0, '.')
from brain_mesh_module import TimeIt
from brain_mesh_module import BrainSection

#==================================================================

class ImageManipulation():

    def __init__(self):

        self.nib_file   = None

        self.nib_is_loaded = False

        self.affine_transformation = None

        self.safe_range = (200, 1000)

        self.main_dir =\
                os.path.join(self.up_one_directory(),\
                'Bmesh')

        self.experimental_data_dir = os.path.join(\
                self.main_dir, 'experimental_data')

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
#==================================================================M

        self.load_nib_affine_transformation()

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
    def store_boundary_data(self):

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

            obj = BrainSection(dir_path,\
                    idx,\
                    gray,\
                    self.affine_transformation)

            obj.extract_boundary()
            obj.store_boundary_pixels()

#==================================================================
    def create_splines(self):

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

            obj.create_splines()

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

        '''
        gray         = gray[y:y+h,x:x+w]
        row_tol = h * 0.1;
        col_tol = w * 0.1;
        bw = 100 < gray
        rows,cols = np.nonzero(bw)
        mask = np.full(rows.size, False)

        for k in range(rows.size):
            r = rows[k]
            c = cols[k]
            distance_to_boundaries = (r,h-r,c,w-c)
            min_idx = np.argmin(distance_to_boundaries)
            value   = distance_to_boundaries[min_idx]
            if min_idx < 2:
                if value < row_tol:
                    mask[k] = True
            else:
                if value < col_tol:
                    mask[k] = True

        cropped_img[rows[mask],cols[mask]] = 0

        erosion_fname = 'erosion.jpeg'
        erosion_fname = os.path.join(\
                dir_path,\
                erosion_fname)

        cv2.imwrite(erosion_fname, cropped_img)
        '''





#==================================================================
    def send_experimental_data_to_model(self):

        number_regex = re.compile(r'[0-9]+')
        z_slice_n_regex = re.compile(r'm_(?P<z_n>[0-9]+)')
        v = np.linalg.solve(\
                self.affine_transformation, np.array([0,0,0,1]))
        print('Center in pixels')
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
            Z-axis parameters
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

            x,y,w,h = cv2.boundingRect(boundary.T)
            print('Model WxH: {:d}x{:d}'.format(w,h))

            x_indices = np.argsort(boundary[0])
            y_indices = np.argsort(boundary[1])
            x_min_pixel = boundary[0,x_indices[0]]
            x_max_pixel = boundary[0,x_indices[-1]]
            y_min_pixel = boundary[1,y_indices[0]]
            y_max_pixel = boundary[1,y_indices[-1]]
            print('min. corner ({:d},{:d}):'.\
                    format(x_min_pixel, y_min_pixel))
            print('max. corner ({:d},{:d}):'.\
                    format(x_max_pixel, y_max_pixel))
 
            delta_x_pixel = x_max_pixel - x_min_pixel + 1            
            delta_y_pixel = y_max_pixel - y_min_pixel + 1            

            model_img_shape = (delta_y_pixel, delta_x_pixel)

            print('Model pixel dimensions')
            print('H x W: {:d} x {:d}'.\
                    format(*model_img_shape))

            model_heigh_to_width_ratio =\
                    model_img_shape[0] /\
                    model_img_shape[1] 

            print('Model H:W')
            print(model_heigh_to_width_ratio)

            exit()

            new_experimental_width =\
                    experimental_img_shape[0] /\
                    model_heigh_to_width_ratio

            new_experimental_width =\
                    np.round(new_experimental_width).astype(int)

            print('New experimental width')
            print(new_experimental_width)

            new_experimental_w_h = (new_experimental_width,\
                    experimental_img_shape[0])

            experimental_img =\
                    cv2.resize(\
                    experimental_img,\
                    model_img_shape[::-1])
                    #new_experimental_w_h)


            experimental_img[\
                    boundary[1]-boundary[1].min(),\
                    boundary[0]-boundary[0].min(),\
                    :] = 0

            new_experimental_fname = 'channel.jpeg'
                    #experimental_basename + '_scaled.jpeg'
            new_experimental_fname = os.path.join(\
                    dir_path,\
                    new_experimental_fname)

            cv2.imwrite(new_experimental_fname,\
                    experimental_img[:,:,:])
            exit()

            '''
            Save model pixel data 
            '''
            boundary_path =\
                    os.path.join(local_slice_dir, boundary_fname)
            np.savetxt(fname)



            print('--------------------------------------')


            
#==================================================================
'''




print(brain_slices_dir)

for (dir_path, dir_names, file_names) in os.walk(brain_slices_dir):

    print('Working with {:s}'.format(dir_path))
    basename = os.path.basename(dir_path)
    obj = number_regex.search(basename)

    if obj is None:
        continue

    idx = int(obj.group(0))
    if (idx < safe_range[0]) or (safe_range[1] < idx):
        continue
    fname = basename + '.txt'

    source_path = os.path.join(brain_slices_txt_dir, fname)
    gray = np.loadtxt(source_path, dtype='uint8')
    BrainSection(dir_path, idx, gray, img.affine)
    print('------------------------------------')



'''

        


