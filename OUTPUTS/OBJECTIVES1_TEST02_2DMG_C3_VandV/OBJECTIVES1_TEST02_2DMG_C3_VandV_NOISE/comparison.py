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
#######################################################################################################
#*************************************************************************************
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES1_TEST02_2DMG_C3_VandV import *
sys.path.remove(inputs_dir)

# Load data from JSON file
with open(f'{case_name}_NOISE_output.json', 'r') as json_file:
    noise_output = json.load(json_file)

# Access dPHI from the loaded data
dPHI1 = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI1"]]
dPHI2 = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI2"]]
dPHI = dPHI1 + dPHI2
dPHI_array = np.array(dPHI)
dPHI_reshaped = dPHI_array.reshape(group, I_max, J_max)

# dFLX CORESIM
dFLX1_CORESIM_array = np.array(dFLX1_CORESIM)
dFLX2_CORESIM_array = np.array(dFLX2_CORESIM)
dFLX1_CORESIM_flattened_array = dFLX1_CORESIM_array.ravel()
dFLX2_CORESIM_flattened_array = dFLX2_CORESIM_array.ravel()
dFLX_CORESIM = [[dFLX1_CORESIM_flattened_array], [dFLX2_CORESIM_flattened_array]]
dFLX_CORESIM_array = np.array(dFLX_CORESIM)
dFLX_CORESIM_reshaped = dFLX_CORESIM_array.reshape(group, I_max, J_max)

# Calculate error and compare
diff_dflx1_CS = np.abs((dFLX1_CORESIM_flattened_array - np.array(dPHI1))/dFLX1_CORESIM_flattened_array) * 100
diff_dflx2_CS = np.abs((dFLX2_CORESIM_flattened_array - np.array(dPHI2))/dFLX2_CORESIM_flattened_array) * 100
diff_dflx_CS = [[diff_dflx1_CS], [diff_dflx2_CS]]
diff_dflx_CS_array = np.array(diff_dflx_CS)
diff_dflx_CS_reshaped = diff_dflx_CS_array.reshape(group, I_max, J_max)

#diff_dflx1_CS_phase = np.angle((dFLX1_CORESIM_flattened_array - np.array(dPHI1))/dFLX1_CORESIM_flattened_array) * 100
#diff_dflx2_CS_phase = np.angle((dFLX2_CORESIM_flattened_array - np.array(dPHI2))/dFLX2_CORESIM_flattened_array) * 100
diff_dflx1_CS_phase = (np.angle(dFLX1_CORESIM_flattened_array) - np.angle(np.array(dPHI1)))/np.angle(dFLX1_CORESIM_flattened_array) * 100
diff_dflx2_CS_phase = (np.angle(dFLX2_CORESIM_flattened_array) - np.angle(np.array(dPHI2)))/np.angle(dFLX2_CORESIM_flattened_array) * 100
diff_dflx_CS_phase = [[diff_dflx1_CS_phase], [diff_dflx2_CS_phase]]
diff_dflx_CS_phase_array = np.array(diff_dflx_CS_phase)
diff_dflx_CS_phase_reshaped = diff_dflx_CS_phase_array.reshape(group, I_max, J_max)

# FLX Sn
dFLX1_Sn_array = np.array(dFLX1_Sn)
dFLX2_Sn_array = np.array(dFLX2_Sn)
dFLX1_Sn_flattened_array = dFLX1_Sn_array.ravel()
dFLX2_Sn_flattened_array = dFLX2_Sn_array.ravel()
dFLX_Sn = [[dFLX1_Sn_flattened_array], [dFLX2_Sn_flattened_array]]
dFLX_Sn_array = np.array(dFLX_Sn)
dFLX_Sn_reshaped = dFLX_Sn_array.reshape(group, I_max, J_max)

# Calculate error and compare
diff_dflx1_Sn = np.abs((dFLX1_Sn_flattened_array - np.array(dPHI1))/dFLX1_Sn_flattened_array) * 100
diff_dflx2_Sn = np.abs((dFLX2_Sn_flattened_array - np.array(dPHI2))/dFLX2_Sn_flattened_array) * 100
diff_dflx_Sn = [[diff_dflx1_Sn], [diff_dflx2_Sn]]
diff_dflx_Sn_array = np.array(diff_dflx_Sn)
diff_dflx_Sn_reshaped = diff_dflx_Sn_array.reshape(group, I_max, J_max)

#diff_dflx1_Sn_phase = np.angle((dFLX1_Sn_flattened_array - np.array(dPHI1))/dFLX1_Sn_flattened_array) * 100
#diff_dflx2_Sn_phase = np.angle((dFLX2_Sn_flattened_array - np.array(dPHI2))/dFLX2_Sn_flattened_array) * 100
diff_dflx1_Sn_phase = (np.angle(dFLX1_Sn_flattened_array) - np.angle(np.array(dPHI1)))/np.angle(dFLX1_Sn_flattened_array) * 100
diff_dflx2_Sn_phase = (np.angle(dFLX2_Sn_flattened_array) - np.angle(np.array(dPHI2)))/np.angle(dFLX2_Sn_flattened_array) * 100
diff_dflx_Sn_phase = [[diff_dflx1_Sn_phase], [diff_dflx2_Sn_phase]]
diff_dflx_Sn_phase_array = np.array(diff_dflx_Sn_phase)
diff_dflx_Sn_phase_reshaped = diff_dflx_Sn_phase_array.reshape(group, I_max, J_max)

#*************************************************************************************
#*************************************************************************************
for g in range(group):
    plot_heatmap(dPHI_reshaped[g], g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1} Magnitude', process_data='magnitude')
    plot_heatmap(dFLX_CORESIM_reshaped[g], g+1, cmap='viridis', varname='dFLX_CORESIM', title=f'2D Plot of dFLX{g+1}_CORESIM Magnitude', process_data='magnitude')
    plot_heatmap(dFLX_Sn_reshaped[g], g+1, cmap='viridis', varname='dFLX_Sn', title=f'2D Plot of dFLX{g+1}_Sn Magnitude', process_data='magnitude')
    plot_heatmap(diff_dflx_Sn_reshaped[g], g+1, cmap='viridis', varname='diff_dflx_Sn', title=f'2D Plot of diff_dflx{g+1}_Sn magnitude in %', process_data='magnitude')
    plot_heatmap(diff_dflx_CS_reshaped[g], g+1, cmap='viridis', varname='diff_dflx_CS', title=f'2D Plot of diff_dflx{g+1}_CS magnitude in %', process_data='magnitude')
    
    plot_heatmap(dPHI_reshaped[g], g+1, cmap='viridis', varname='dPHI', title=f'2D Plot of dPHI{g+1} Phase', process_data='phase')
    plot_heatmap(dFLX_CORESIM_reshaped[g], g+1, cmap='viridis', varname='dFLX_CORESIM', title=f'2D Plot of dFLX{g+1}_CORESIM Phase', process_data='phase')
    plot_heatmap(dFLX_Sn_reshaped[g], g+1, cmap='viridis', varname='dFLX_Sn', title=f'2D Plot of dFLX{g+1}_Sn Phase', process_data='phase')

    plt.clf()
    plt.imshow(diff_dflx_Sn_phase_reshaped[g], cmap='viridis', interpolation='nearest')
    plt.colorbar(label=f'diff_dflx{g}_Sn_phase')  # Add color bar to show scale
    plt.title(f'2D Plot of diff_dflx{g+1}_Sn Phase in %')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    if noise_pos != 0:
        if type_noise != 0:
            plt.savefig(f'{case_name}_NOISE_diff_dflx_Sn_phase_G{g+1}.png')
    else:
        plt.savefig(f'{case_name}_NOISE_diff_dflx_Sn_phase_G{g+1}.png')

    plt.clf()
    plt.imshow(diff_dflx_CS_phase_reshaped[g], cmap='viridis', interpolation='nearest')
    plt.colorbar(label=f'diff_dflx{g}_CS_phase')  # Add color bar to show scale
    plt.title(f'2D Plot of diff_dflx{g+1}_CS Phase in %')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    if noise_pos != 0:
        if type_noise != 0:
            plt.savefig(f'{case_name}_NOISE_diff_dflx_CS_phase_G{g+1}.png')
    else:
        plt.savefig(f'{case_name}_NOISE_diff_dflx_CS_phase_G{g+1}.png')
