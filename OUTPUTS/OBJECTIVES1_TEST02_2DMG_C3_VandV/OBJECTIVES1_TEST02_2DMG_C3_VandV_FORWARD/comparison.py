import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#######################################################################################################
def plot_heatmap(data, g, cmap='viridis', varname=None, title=None):
    plt.clf()

    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

    plt.colorbar(label=f'{varname}{g}')
    if title:
        plt.title(title)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    
    x_ticks = np.linspace(x.min(), x.max(), num=10)
    y_ticks = np.linspace(y.min(), y.max(), num=10)
    plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
    plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

    filename = f'{case_name}_FORWARD_{varname}_G{g}.png'
    plt.savefig(filename)
    plt.close()

    return filename
#######################################################################################################
#*************************************************************************************
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES1_TEST02_2DMG_C3_VandV import *
sys.path.remove(inputs_dir)

# Load data from JSON file
with open(f'{case_name}_FORWARD_output.json', 'r') as json_file:
    forward_output = json.load(json_file)

# Access keff and PHI from the loaded data
keff = forward_output["keff"]
PHI1 = forward_output["PHI1_FORWARD"]
PHI2 = forward_output["PHI2_FORWARD"]
PHI = PHI1 + PHI2
PHI_array = np.array(PHI)
PHI_reshaped = PHI_array.reshape(group, I_max, J_max)

# FLX CORESIM
FLX1_CORESIM_array = np.array(FLX1_CORESIM)
FLX2_CORESIM_array = np.array(FLX2_CORESIM)
FLX1_CORESIM_flattened_array = FLX1_CORESIM_array.ravel()
FLX2_CORESIM_flattened_array = FLX2_CORESIM_array.ravel()
FLX_CORESIM = [[FLX1_CORESIM_flattened_array], [FLX2_CORESIM_flattened_array]]
FLX_CORESIM_array = np.array(FLX_CORESIM)
FLX_CORESIM_reshaped = FLX_CORESIM_array.reshape(group, I_max, J_max)

# FLX Sn
FLX1_SN_array = np.array(FLX1_Sn)
FLX2_SN_array = np.array(FLX2_Sn)
FLX1_SN_flattened_array = FLX1_SN_array.ravel()
FLX2_SN_flattened_array = FLX2_SN_array.ravel()
FLX_SN = [[FLX1_SN_flattened_array], [FLX2_SN_flattened_array]]
FLX_SN_array = np.array(FLX_SN)
FLX_SN_reshaped = FLX_SN_array.reshape(group, I_max, J_max)

# Calculate error and compare
diff_flx1_CS = np.abs((FLX1_CORESIM_flattened_array - np.array(PHI1))/FLX1_CORESIM_flattened_array) * 100
diff_flx2_CS = np.abs((FLX2_CORESIM_flattened_array - np.array(PHI2))/FLX2_CORESIM_flattened_array) * 100
diff_flx_CS = [[diff_flx1_CS], [diff_flx2_CS]]
diff_flx_CS_array = np.array(diff_flx_CS)
diff_flx_CS_reshaped = diff_flx_CS_array.reshape(group, I_max, J_max)

# Calculate error and compare
diff_flx1_SN = np.abs((FLX1_SN_flattened_array - np.array(PHI1))/FLX1_SN_flattened_array) * 100
diff_flx2_SN = np.abs((FLX2_SN_flattened_array - np.array(PHI2))/FLX2_SN_flattened_array) * 100
diff_flx_SN = [[diff_flx1_SN], [diff_flx2_SN]]
diff_flx_SN_array = np.array(diff_flx_SN)
diff_flx_SN_reshaped = diff_flx_SN_array.reshape(group, I_max, J_max)
#*************************************************************************************
#*************************************************************************************
for g in range(group):
    plot_heatmap(PHI_reshaped[g], g+1, cmap='viridis', varname='PHI', title=f'2D Plot of PHI{g+1}')
    plot_heatmap(FLX_CORESIM_reshaped[g], g+1, cmap='viridis', varname='FLX_CORESIM', title=f'2D Plot of FLX{g+1}_CORESIM')
    plot_heatmap(FLX_SN_reshaped[g], g+1, cmap='viridis', varname='FLX_SN', title=f'2D Plot of FLX{g+1}_SN')
    plot_heatmap(diff_flx_CS_reshaped[g], g+1, cmap='viridis', varname='diff_flx_CS', title=f'2D Plot of Relative Difference group {g+1} in %\n Simulator vs CORE SIM+')
    plot_heatmap(diff_flx_SN_reshaped[g], g+1, cmap='viridis', varname='diff_flx_SN', title=f'2D Plot of Relative Difference group {g+1} in %\n Simulator vs S8')
