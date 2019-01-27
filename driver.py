import sys
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

sys.path.insert(0, './modules')
from brain_mesh_module import TimeIt
from brain_mesh_module import BrainSection
from image_manipulation_module import ImageManipulation

obj = ImageManipulation()
obj.send_experimental_data_to_model()
#obj.store_boundary_data()

#x  = obj.map_experimental_z_n_to_model_z_n(2)
#print(obj.map_model_z_n_to_mm(1000))

