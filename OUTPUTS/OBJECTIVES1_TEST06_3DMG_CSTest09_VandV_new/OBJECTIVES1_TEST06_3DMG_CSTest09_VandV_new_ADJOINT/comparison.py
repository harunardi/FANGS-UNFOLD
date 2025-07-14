import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from matplotlib import cm
from PIL import Image

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#######################################################################################################
def plot_heatmap_3D(data, g, z, x, y, cmap='viridis', varname=None, title=None, case_name=None, process_data=None, solve=None):
    plt.clf()
    if process_data == 'magnitude':
        data = np.abs(data)  # Compute magnitude
    elif process_data == 'phase':
        data_rad = np.angle(data)  # Compute phase
        data = np.degrees(data_rad)  # Compute phase

    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

    if process_data == 'magnitude':
        plt.colorbar(label=f'{varname}{g}')  # Add color bar to show scale
    elif process_data == 'phase':
        plt.colorbar(label=f'{varname}{g}_deg')  # Add color bar to show scale

    if title:
        plt.title(title)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    
    # Define ticks every 10 cm
    x_ticks = np.linspace(x.min(), x.max(), num=10)
    y_ticks = np.linspace(y.min(), y.max(), num=10)
    plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
    plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

    filename = f'{case_name}_{solve}_{varname}_{process_data}_G{g}_Z{z}.png'
    plt.savefig(filename)
    plt.close()

    return filename
#######################################################################################################
#*************************************************************************************
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES1_TEST06_3DMG_CSTest09_VandV_new import *
sys.path.remove(inputs_dir)

# Load data from JSON file
with open(f'{case_name}_ADJOINT_output.json', 'r') as json_file:
    forward_output = json.load(json_file)

# Access keff and PHI from the loaded data
keff = forward_output["keff"]
PHI1 = forward_output["PHI1_ADJOINT"]
PHI2 = forward_output["PHI2_ADJOINT"]
PHI = PHI1 + PHI2
PHI_array = np.array(PHI)
PHI_reshaped = PHI_array.reshape(group, K_max, J_max, I_max)

# FLX CORESIM
FLX1_CORESIM_array = np.array(FLX1_ADJ_CORESIM)
FLX2_CORESIM_array = np.array(FLX2_ADJ_CORESIM)
FLX1_CORESIM_flattened_array = FLX1_CORESIM_array.ravel()
FLX2_CORESIM_flattened_array = FLX2_CORESIM_array.ravel()
FLX_CORESIM = [[FLX1_CORESIM_flattened_array], [FLX2_CORESIM_flattened_array]]
FLX_CORESIM_array = np.array(FLX_CORESIM)
FLX_CORESIM_reshaped = FLX_CORESIM_array.reshape(group, K_max, J_max, I_max)

# Calculate error and compare
diff_flx1_CS = np.abs((FLX1_CORESIM_flattened_array - np.array(PHI1))/FLX1_CORESIM_flattened_array) * 100
diff_flx2_CS = np.abs((FLX2_CORESIM_flattened_array - np.array(PHI2))/FLX2_CORESIM_flattened_array) * 100
diff_flx_CS = [[diff_flx1_CS], [diff_flx2_CS]]
diff_flx_CS_array = np.array(diff_flx_CS)
diff_flx_CS_reshaped = diff_flx_CS_array.reshape(group, K_max, J_max, I_max)

#*************************************************************************************
for g in range(group):
    image_files = []
    for k in range(K_max):
        filename_PHI = plot_heatmap_3D(FLX_CORESIM_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='FLX_ADJ', title=f'2D Plot of FLX{g+1}_ADJ, Z={k+1}', case_name=case_name, process_data='magnitude', solve='ADJOINT')
        image_files.append(filename_PHI)

    # Create a GIF from the saved images
    gif_filename_PHI = f'{case_name}_ADJOINT_FLX_animation_G{g+1}.gif'

    # Open images and save as GIF
    images_PHI = [Image.open(img) for img in image_files]
    images_PHI[0].save(gif_filename_PHI, save_all=True, append_images=images_PHI[1:], duration=300, loop=0)
    print(f"GIF saved as {gif_filename_PHI}")

for g in range(group):
    image_files = []
    for k in range(K_max):
        filename_PHI = plot_heatmap_3D(diff_flx_CS_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='diff_PHI', title=f'2D Plot of diff_PHI{g+1}_ADJ, Z={k+1} in %', case_name=case_name, process_data='magnitude', solve='ADJOINT')
        image_files.append(filename_PHI)

    # Create a GIF from the saved images
    gif_filename_PHI = f'{case_name}_ADJOINT_diff_PHI_animation_G{g+1}.gif'

    # Open images and save as GIF
    images_PHI = [Image.open(img) for img in image_files]
    images_PHI[0].save(gif_filename_PHI, save_all=True, append_images=images_PHI[1:], duration=300, loop=0)
    print(f"GIF saved as {gif_filename_PHI}")
