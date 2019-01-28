import sys
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import cv2

#sys.path.insert(0, '.')
from timing_module import TimeIt
from brain_slice_module import BrainSection

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
        self.experimental_center_mm_in_model = np.array([2,0]) 
        
        '''
        From top of the brain (bregma) to site of injection
        '''
        self.depth_of_injection_in_mm = 3.0

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
    def get_experimental_center_in_model_pixel_xy(self, z_mm):

        c_mm = np.array([\
                self.experimental_center_mm_in_model[0],\
                z_mm,\
                self.experimental_center_mm_in_model[1],\
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
            c = self.get_experimental_center_in_model_pixel_xy(\
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

            print('New experimental width')
            print(new_experimental_width)

            experimental_img_shape = (\
                    experimental_img_shape[0],\
                    new_experimental_width,\
                    )
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
                    fmt='%d');

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
                    self.experimental_center_mm_in_model[0]
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
            Store red channel
            Recall that the output format is BGR
            '''
            channel = experimental_img[:,:,2]
            channel_fname = 'channel.jpeg'
            channel_fname = os.path.join(\
                    dir_path,\
                    channel_fname)

            cv2.imwrite(channel_fname,\
                    channel)



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


            '''
            Define map from model to experimental
            '''
            def model_to_experimental_value(x,y):
                '''
                Change to coordinates wrt the experimental center
                '''
                xc = x - self.experimental_center_mm_in_model[0]
                yc = y - self.experimental_center_mm_in_model[1]
                xc *=  experimental_img_shape[1]/model_width_mm
                yc *= -experimental_img_shape[0]/model_height_mm
                xc += experimental_center_pixel[0]
                yc += experimental_center_pixel[1]

                return (xc, yc)



            brain_slice = BrainSection(\
                    local_slice_dir,\
                    model_z_n,\
                    None,\
                    self.affine_transformation,\
                    )

            brain_slice.load_splines()
            brain_slice.test_spline_slice()

            exit()




            
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

        


