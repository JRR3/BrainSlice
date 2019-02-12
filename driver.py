import sys
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

#==================================================================

sys.path.insert(0, './modules')
from timing_module import TimeIt
from brain_slice_module import BrainSection
from image_manipulation_module import ImageManipulation
from fem_module import FEMSimulation

#==================================================================

#obj = ImageManipulation()
#obj.get_radial_plot()

#obj.store_boundary_data()
#obj.generate_experimental_data_essentials()
#obj.load_experimental_data_essentials()
#obj.plot_interpolated_slices()
#obj.plot_sphere()

obj = FEMSimulation()
obj.run()

#x  = obj.map_experimental_z_n_to_model_z_n(2)
#print(obj.map_model_z_n_to_mm(1000))

