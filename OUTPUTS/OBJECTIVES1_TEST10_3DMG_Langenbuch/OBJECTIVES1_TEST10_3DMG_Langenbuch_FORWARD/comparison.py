import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from PIL import Image

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#######################################################################################################
def plot_heatmap_3D(data, g, z, x, y, cmap='viridis', varname=None, case_name=None, title=None):
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

    filename = f'{case_name}_FORWARD_{varname}_G{g}_Z{z}.png'
    plt.savefig(filename)
    plt.close()

    return filename

def convert_index_3D_rect(D, I_max, J_max, K_max):
    conv = [0] * (I_max * J_max * K_max)
    tmp_conv = 0
    for k in range(K_max):
        for j in range(J_max):
            for i in range(I_max):
                if D[0][k][j][i] != 0:
                    tmp_conv += 1
                    m = k * (I_max * J_max) + j * I_max + i
                    conv[m] = tmp_conv
    return conv

#######################################################################################################
#*************************************************************************************
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES1_TEST10_3DMG_Langenbuch import *
sys.path.remove(inputs_dir)

print(len(FLX), len(FLX[0]), len(FLX[0][0]))

# FLX FEMFFUSION
I_max_FLX = 11
J_max_FLX = 11
K_max_FLX = 10
max_FLX = np.max(FLX)
print("max_FLX", max_FLX)

for g in range(group):
    for k in range(K_max_FLX):
        for j in range(J_max_FLX):
            for i in range(I_max_FLX):
                m = j*I_max_FLX + i
                FLX[g][k][m] = FLX[g][k][m] / max_FLX
                if FLX[g][k][m] == 0:
                    FLX[g][k][m] = np.nan

FLX_reshaped = np.array(FLX).reshape(group, K_max_FLX, J_max_FLX, I_max_FLX)
for g in range(group):
    image_files = []
    for k in range(K_max):
        filename_FLX = plot_heatmap_3D(FLX_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='FLX', case_name=case_name, title=f'2D Plot of FLX{g+1}, Z={k+1}')
        image_files.append(filename_FLX)
    # Create a GIF from the saved images
    gif_filename_FLX = f'FLX_animation_G{g+1}.gif'
    # Open images and save as GIF
    images_FLX = [Image.open(img) for img in image_files]
    images_FLX[0].save(gif_filename_FLX, save_all=True, append_images=images_FLX[1:], duration=300, loop=0)
    print(f"GIF saved as {gif_filename_FLX}")

#*************************************************************************************
# Load data from JSON file
with open(f'{case_name}_FORWARD_output.json', 'r') as json_file:
    forward_output = json.load(json_file)

# Access keff and PHI from the loaded data
keff = forward_output["keff"]
PHI1 = forward_output["PHI1_FORWARD"]
PHI2 = forward_output["PHI2_FORWARD"]
PHI = PHI1 + PHI2
PHI = np.reshape(PHI, (group, N))

POWER = np.zeros_like(PHI)
for g in range(group):
    for n in range(len(PHI1)):
        POWER[g][n] = PHI[g][n] * 1.0

new_size = 11
cells_per_block = 10

# Reshape POWER to (2, 17, 17, 5, 5) for easier summing
POWER_reshaped = POWER.reshape(group, K_max, J_max, I_max)

# Sum over each 10x10 block
POWER_collapsed = np.zeros((group, K_max_FLX, J_max_FLX, I_max_FLX))
for g in range(2):
    for k in range(K_max):
        for j in range(new_size):
            for i in range(new_size):
                block = POWER_reshaped[g, k, j*cells_per_block:(j+1)*cells_per_block,
                                            i*cells_per_block:(i+1)*cells_per_block]
                POWER_collapsed[g, k, j, i] = np.sum(block)
dV_asmb = 23.1226*23.1226

PHI_reshaped = POWER_collapsed / dV_asmb
max_PHI = np.nanmax(PHI_reshaped)
for g in range(group):
    for k in range(K_max_FLX):
        for j in range(J_max_FLX):
            for i in range(I_max_FLX):
                PHI_reshaped[g][k][j][i] = PHI_reshaped[g][k][j][i] / max_PHI
                if PHI_reshaped[g][k][j][i] == 0:
                    PHI_reshaped[g][k][j][i] = np.nan

for g in range(group):
    image_files = []
    for k in range(K_max):
        filename_PHI = plot_heatmap_3D(PHI_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='PHI_norm', case_name=case_name, title=f'2D Plot of PHI{g+1} Normalized, Z={k+1}')
        image_files.append(filename_PHI)
    # Create a GIF from the saved images
    gif_filename_PHI = f'PHI_normalized_animation_G{g+1}.gif'
    # Open images and save as GIF
    images_PHI = [Image.open(img) for img in image_files]
    images_PHI[0].save(gif_filename_PHI, save_all=True, append_images=images_PHI[1:], duration=300, loop=0)
    print(f"GIF saved as {gif_filename_PHI}")

#*************************************************************************************
# Calculate error and compare
diff_flx_reshaped = np.zeros((group, K_max_FLX, J_max_FLX, I_max_FLX))
for g in range(group):
    for k in range(K_max_FLX):
        for j in range(J_max_FLX):
            for i in range(I_max_FLX):
                if FLX_reshaped[g][k][j][i] != 0:
                    diff_flx_reshaped[g][k][j][i] = np.abs((FLX_reshaped[g][k][j][i] - PHI_reshaped[g][k][j][i]) / FLX_reshaped[g][k][j][i]) * 100
                else:
                    diff_flx_reshaped[g][k][j][i] = np.nan

for g in range(group):
    image_files = []
    for k in range(K_max):
        filename_diff_flx = plot_heatmap_3D(diff_flx_reshaped[g, k, :, :], g+1, k+1, x, y, cmap='viridis', varname='diff_flx', case_name=case_name, title=f'2D Plot of diff_flx{g+1}, Z={k+1}')
        image_files.append(filename_diff_flx)
    # Create a GIF from the saved images
    gif_filename_diff_flx = f'diff_flx_animation_G{g+1}.gif'
    # Open images and save as GIF
    images_diff_flx = [Image.open(img) for img in image_files]
    images_diff_flx[0].save(gif_filename_diff_flx, save_all=True, append_images=images_diff_flx[1:], duration=300, loop=0)
    print(f"GIF saved as {gif_filename_diff_flx}")




#for g in range(group):
#    plot_heatmap(FLX_reshaped[g], g+1, cmap='viridis', varname='FLX_FEMFFUSION', title=f'2D Plot of FLX{g+1}_FEMFFUSION')
#    plot_heatmap(PHI_reshaped[g], g+1, cmap='viridis', varname='PHI_normalized', title=f'2D Plot of PHI{g+1}_NORMALIZED')
#    plot_heatmap(diff_flx_reshaped[g], g+1, cmap='viridis', varname='diff_flx', title=f'2D Plot of Relative Difference group {g+1} in %\n Simulator vs FEMFFUSION')
#