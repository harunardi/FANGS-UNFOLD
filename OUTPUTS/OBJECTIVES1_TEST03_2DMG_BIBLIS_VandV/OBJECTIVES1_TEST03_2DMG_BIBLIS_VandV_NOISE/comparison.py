import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#######################################################################################################
def plot_heatmap(data, g, cmap='viridis', varname=None, title=None, process_data=None):
    plt.clf()
    if process_data == 'magnitude':
        data = np.abs(data)  # Compute magnitude
    elif process_data == 'phase':
        data_rad = np.angle(data)  # Compute phase
        data = np.degrees(data_rad)  # Compute phase
    
    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

    if process_data == 'magnitude':
        plt.colorbar(label=f'{varname}{g}_mag')  # Add color bar to show scale
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

    if noise_pos != 0:
        if type_noise != 0:
            filename = f'{case_name}_NOISE_{varname}_{process_data}_G{g}.png'
    else:
        filename = f'{case_name}_NOISE_{varname}_{process_data}_G{g}.png'
    plt.savefig(filename)
    plt.close()

    return filename

def convert_index_2D_rect(D, I_max, J_max):
    conv = [0] * (I_max*J_max)
    tmp_conv = 0
    for j in range(J_max):  
        for i in range(I_max):
            if D[0][j][i] != 0:
                tmp_conv += 1
                m = j * I_max + i
                conv[m] = tmp_conv
    return conv

#######################################################################################################
#*************************************************************************************
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES1_TEST03_2DMG_BIBLIS_VandV import *
sys.path.remove(inputs_dir)

# FLX FEMFFUSION
I_max_FLX = 17
J_max_FLX = 17
dFLX_reshaped = [dFLX1, dFLX2]
max_FLX = np.max(dFLX_reshaped)

for g in range(group):
    for j in range(J_max_FLX):
        for i in range(I_max_FLX):
            dFLX_reshaped[g][j][i] = dFLX_reshaped[g][j][i] / max_FLX
            if dFLX_reshaped[g][j][i] == 0:
                dFLX_reshaped[g][j][i] = np.nan

# Load data from JSON file
with open(f'{case_name}_NOISE_output.json', 'r') as json_file:
    noise_output = json.load(json_file)

# Access dPHI from the loaded data
dPHI1 = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI1"]]
dPHI2 = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI2"]]
dPHI = dPHI1 + dPHI2
dPHI = np.reshape(dPHI, (group, N))

POWER = np.zeros_like(dPHI)
for g in range(group):
    for n in range(len(dPHI1)):
        POWER[g][n] = dPHI[g][n] * 1.0 * (dx*dy)

new_size = 17
cells_per_block = 5

# Reshape POWER to (2, 17, 17, 5, 5) for easier summing
POWER_reshaped = POWER.reshape(group, J_max, I_max)

# Sum over each 5x5 block
POWER_collapsed = np.zeros((group, J_max_FLX, I_max_FLX))
for g in range(2):
    for j in range(new_size):
        for i in range(new_size):
            block = POWER_reshaped[g, j*cells_per_block:(j+1)*cells_per_block,
                                        i*cells_per_block:(i+1)*cells_per_block]
            POWER_collapsed[g, j, i] = np.sum(block)

dV_asmb = 23.1226*23.1226

dPHI_reshaped = POWER_collapsed / dV_asmb
max_dPHI = np.nanmax(dPHI_reshaped)
for g in range(group):
    for j in range(J_max_FLX):
        for i in range(I_max_FLX):
            dPHI_reshaped[g][j][i] = dPHI_reshaped[g][j][i] / max_dPHI
            if dFLX_reshaped[g][j][i] == 0:
                dPHI_reshaped[g][j][i] = np.nan

# Calculate error and compare
diff_flx_reshaped = np.zeros((group, J_max_FLX, I_max_FLX))
for g in range(group):
    for j in range(J_max_FLX):
        for i in range(I_max_FLX):
            if dFLX_reshaped[g][j][i] != 0:
                diff_flx_reshaped[g][j][i] = np.abs((dFLX_reshaped[g][j][i] - dPHI_reshaped[g][j][i]) / dFLX_reshaped[g][j][i]) * 100
            else:
                diff_flx_reshaped[g][j][i] = np.nan

#*************************************************************************************
for g in range(group):
    plot_heatmap(dFLX_reshaped[g], g+1, cmap='viridis', varname='dFLX', title=f'2D Plot of dFLX{g+1} Magnitude', process_data='magnitude')
    plot_heatmap(dFLX_reshaped[g], g+1, cmap='viridis', varname='dFLX', title=f'2D Plot of dFLX{g+1} Phase', process_data='phase')
    plot_heatmap(dPHI_reshaped[g], g+1, cmap='viridis', varname='dPHI_NORMALIZED', title=f'2D Plot of dPHI{g+1}_NORMALIZED Magnitude', process_data='magnitude')
    plot_heatmap(dPHI_reshaped[g], g+1, cmap='viridis', varname='dPHI_NORMALIZED', title=f'2D Plot of dPHI{g+1}_NORMALIZED Phase', process_data='phase')
    plot_heatmap(diff_flx_reshaped[g], g+1, cmap='viridis', varname='diff_dFLX', title=f'2D Plot of Relative Difference group {g+1} in %\n Simulator vs FEMFFUSION Magnitude', process_data='magnitude')
