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
FLX_reshaped = [FLX1, FLX2]
max_FLX = np.max(FLX_reshaped)

for g in range(group):
    for j in range(J_max_FLX):
        for i in range(I_max_FLX):
            FLX_reshaped[g][j][i] = FLX_reshaped[g][j][i] / max_FLX
            if FLX_reshaped[g][j][i] == 0:
                FLX_reshaped[g][j][i] = np.nan

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

PHI_reshaped = POWER_collapsed / dV_asmb
max_PHI = np.nanmax(PHI_reshaped)
for g in range(group):
    for j in range(J_max_FLX):
        for i in range(I_max_FLX):
            PHI_reshaped[g][j][i] = PHI_reshaped[g][j][i] / max_PHI
            if FLX_reshaped[g][j][i] == 0:
                PHI_reshaped[g][j][i] = np.nan

# Calculate error and compare
diff_flx_reshaped = np.zeros((group, J_max_FLX, I_max_FLX))
for g in range(group):
    for j in range(J_max_FLX):
        for i in range(I_max_FLX):
            if FLX_reshaped[g][j][i] != 0:
                diff_flx_reshaped[g][j][i] = np.abs((FLX_reshaped[g][j][i] - PHI_reshaped[g][j][i]) / FLX_reshaped[g][j][i]) * 100
            else:
                diff_flx_reshaped[g][j][i] = np.nan

#*************************************************************************************
for g in range(group):
    plot_heatmap(FLX_reshaped[g], g+1, cmap='viridis', varname='FLX_FEMFFUSION', title=f'2D Plot of FLX{g+1}_FEMFFUSION')
    plot_heatmap(PHI_reshaped[g], g+1, cmap='viridis', varname='PHI_normalized', title=f'2D Plot of PHI{g+1}_NORMALIZED')
    plot_heatmap(diff_flx_reshaped[g], g+1, cmap='viridis', varname='diff_flx', title=f'2D Plot of Relative Difference group {g+1} in %\n Simulator vs FEMFFUSION')
