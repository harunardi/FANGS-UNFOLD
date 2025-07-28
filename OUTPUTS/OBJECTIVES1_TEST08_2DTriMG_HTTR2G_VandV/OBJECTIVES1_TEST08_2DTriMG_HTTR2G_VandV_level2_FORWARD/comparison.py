import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from collections import defaultdict
import json
import os
import sys
import scipy.special as sp
from scipy.special import j0  # Bessel function J_0

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

# Function to convert 2D hexagonal indexes
def convert_hexx(I_max, J_max, D):
    conv_hexx = [0] * (I_max*J_max)
    tmp_conv = 0
    for j in range(J_max):  
        for i in range(I_max):
            if D[0][j][i] != 0:
                tmp_conv += 1
                m = j * I_max + i
                conv_hexx[m] = tmp_conv

    return conv_hexx

# Function to convert 2D hexagonal indexes to triangular
def convert_tri(I_max, J_max, conv_hexx, level):
    """
    Divide the hexagons into 6 triangles and create a list with numbered index of the variable. 
    This is used to reorder the 2D variable into a column vector.
 
    Parameters
    ----------
    I_max : int
            The size of the column of the list.
    J_max : int
            The size of the row of the list.
    D : list
        The 2D list of diffusion coefficient
 
    Returns
    -------
    conv_tri : list
               The list with numbered index based on the 2D list input.
    D_hexx : list
             The expanded list of diffusion coefficient (triangles)   
    """
    n = 6 * (4 ** (level - 1))

    conv_tri = [0] * I_max * J_max * n
    for j in range(J_max):
        for i in range(I_max):
            m = j * I_max + i
            if conv_hexx[m] != 0:
                for k in range(n):
                    conv_tri[m * n + k] = conv_hexx[m] * n - (n - k - 1)

    conv_hexx_ext = [0] * I_max * J_max * n
    for j in range(J_max):
        for i in range(I_max):
            m = j * I_max + i
            if conv_hexx[m] != 0:
                for k in range(n):
                    conv_hexx_ext[m * n + k] = conv_hexx[m]

    return conv_tri, conv_hexx_ext

def expand_XS_hexx(group, J_max, I_max, XS, level):
    n = 6 * (4 ** (level - 1))

    XS_temp = np.reshape(XS, (group, I_max, J_max))
    XS_hexx = [[0] * (I_max * J_max * n) for _ in range(group)]

    for g in range(group):
        for j in range(J_max):
            for i in range(I_max):
                m = j * I_max + i
                for k in range(n):
                    XS_hexx[g][m * n + k] = XS_temp[g][j][i]

    return XS_hexx

#################################################################
# Change the directory to SRC_all
os.chdir('../../../INPUTS')
sys.path.append(os.getcwd())

from TASK1_TEST18_2DTriMG_HTTR2G_VandV import *

os.chdir('../OUTPUTS/TASK1_TEST18_2DTriMG_HTTR2G_VandV')
sys.path.append(os.getcwd())

from SPECIAL_PLOT_TRIANGULAR_level1 import *
from SPECIAL_PLOT_TRIANGULAR_level2 import *
from SPECIAL_PLOT_TRIANGULAR_level3 import *
from SPECIAL_PLOT_TRIANGULAR_level4 import *

os.chdir(f'TASK1_TEST18_2DTriMG_HTTR2G_VandV_level{level}_FORWARD')
sys.path.append(os.getcwd())

output_dir = f'OUTPUTS/{input_name}'

# Make neighbor indexes
conv_hexx = convert_hexx(I_max, J_max, D)
conv_tri = convert_tri(I_max, J_max, conv_hexx, level)

# Load data from JSON file
with open(f'{case_name}_FORWARD_output.json', 'r') as json_file:
    forward_output = json.load(json_file)

# Access keff and PHI from the loaded data
keff = forward_output["keff"]
PHI1 = forward_output["PHI1"]
PHI2 = forward_output["PHI2"]
PHI = PHI1 + PHI2
PHI_array = np.array(PHI)
PHI_reshaped = PHI_array.reshape(group, N_hexx)

FLX1 = FLX[0]
FLX2 = FLX[1]

# Combine both lists to find the overall maximum value
FLX_combined = FLX1 + FLX2
max_FLX = max(FLX_combined)

# Normalize FLX1 and FLX2 by the overall maximum value
FLX1_normalized = [x / max_FLX for x in FLX1]
FLX2_normalized = [x / max_FLX for x in FLX2]

FLX_reshaped = [FLX1, FLX2]
FLX_new_reshaped = expand_XS_hexx(group, J_max, I_max, FLX_reshaped, level)
FLX1_new_flattened_array = np.array(FLX_new_reshaped[0]).ravel()
FLX2_new_flattened_array = np.array(FLX_new_reshaped[1]).ravel()
diff_flx1 = np.abs(np.array(PHI1) - FLX1_new_flattened_array)/np.abs(FLX1_new_flattened_array) * 100 # in %
diff_flx2 = np.abs(np.array(PHI2) - FLX2_new_flattened_array)/np.abs(FLX2_new_flattened_array) * 100 # in %
diff_flx = [[diff_flx1], [diff_flx2]]
diff_flx_array = np.array(diff_flx)
max_diff = np.max(diff_flx_array)
diff_flx_reshaped = diff_flx_array.reshape(group, N_hexx)

##*************************************************************************************

for g in range(group):
    if level == 1:
        plot_triangular_level1(PHI_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='PHI', title=f'2D Plot of PHI{g+1} Hexx', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
        plot_triangular_level1(FLX_new_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='FLX', title=f'2D Plot of FLX{g+1} Hexx', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
        plot_triangular_level1(diff_flx_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='diff', title=f'2D Plot of diff FLX{g+1} Hexx (in %)', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
    elif level == 2:
        plot_triangular_level2(PHI_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='PHI', title=f'2D Plot of PHI{g+1} Hexx', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
        plot_triangular_level2(FLX_new_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='FLX', title=f'2D Plot of FLX{g+1} Hexx', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
        plot_triangular_level2(diff_flx_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='diff', title=f'2D Plot of diff FLX{g+1} Hexx (in %)', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
    elif level == 3:
        plot_triangular_level3(PHI_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='PHI', title=f'2D Plot of PHI{g+1} Hexx', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
        plot_triangular_level3(FLX_new_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='FLX', title=f'2D Plot of FLX{g+1} Hexx', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
        plot_triangular_level3(diff_flx_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='diff', title=f'2D Plot of diff FLX{g+1} Hexx (in %)', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
    elif level == 4:
        plot_triangular_level4(PHI_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='PHI', title=f'2D Plot of PHI{g+1} Hexx', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
        plot_triangular_level4(FLX_new_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='FLX', title=f'2D Plot of FLX{g+1} Hexx', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
        plot_triangular_level4(diff_flx_reshaped[g], h, I_max, J_max, g+1, level, cmap='viridis', varname='diff', title=f'2D Plot of diff FLX{g+1} Hexx (in %)', case_name=case_name, output_dir=output_dir, solve="forward", process_data="magnitude")
