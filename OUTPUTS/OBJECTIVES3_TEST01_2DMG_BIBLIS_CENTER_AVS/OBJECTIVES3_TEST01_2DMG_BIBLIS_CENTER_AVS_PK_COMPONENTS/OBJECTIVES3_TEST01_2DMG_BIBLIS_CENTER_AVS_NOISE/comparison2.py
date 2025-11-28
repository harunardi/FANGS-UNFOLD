import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from collections import defaultdict
import json
import os
import sys
import scipy.special as sp
from scipy.special import jv, yv, kv, iv

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

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

def plot_1D_centerline_y0_2(PHI1g, PHI2g, conv, I_max, J_max, dx, dy, g, varname=None, process_data=None):

    if process_data == 'magnitude':
        PHI1g = np.abs(PHI1g)  # Compute magnitude
        PHI2g = np.abs(PHI2g)  # Compute magnitude
    elif process_data == 'phase':
        PHI1g = np.degrees(np.angle(PHI1g))  # Convert rad to deg
        PHI2g = np.degrees(np.angle(PHI2g))  # Convert rad to deg
    else:
        pass

    x_coords = np.zeros(I_max * J_max)
    y_coords = np.zeros(I_max * J_max)


    n = 0
    for j in range(J_max):
        for i in range(I_max):
            x_coords[n] = i * dx + dx / 2
            y_coords[n] = j * dy + dy / 2
            n += 1

    x_center = (x_coords.min() + x_coords.max()) / 2
    y_center = (y_coords.min() + y_coords.max()) / 2

    x = x_coords - x_center
    y = y_coords - y_center

    conv_array = np.array(conv)
    PHI1g_temp = np.zeros(max(conv) * group)
    PHI2g_temp = np.zeros(max(conv) * group)
    for g1 in range(group):
        PHI1_indices = g1 * max(conv) + (conv_array - 1)
        PHI1g_temp[PHI1_indices] = PHI1g
        PHI2g_temp[PHI1_indices] = PHI2g
    
    print(np.nanmax(PHI1g_temp), np.nanmax(PHI2g_temp))
    tolerance = 1.5e-0  # Define a small tolerance for floating point comparisons
    distance_flux1_map = defaultdict(list)
    distance_flux2_map = defaultdict(list)

    for n in range(len(x)):
        if abs(y[n]) < tolerance:
            xc = x[n]
            if PHI1g_temp[n] != 0:
                distance_flux1_map[xc].append(PHI1g[n])
            if PHI2g_temp[n] != 0:
                distance_flux2_map[xc].append(PHI2g[n])

    # Extract maximum flux at each signed distance
    unique_distances = sorted(distance_flux1_map.keys())
    flux1_values = [max(distance_flux1_map[d]) for d in unique_distances]
    flux2_values = [max(distance_flux2_map[d]) for d in unique_distances]

    # Plot distance vs max flux values
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot primary y-axis (left)
    ax1.plot(unique_distances, flux1_values, 'b', markersize=5, label=f'dPHI{g}_pk')
    ax1.plot(unique_distances, flux2_values, 'r', markersize=5, label=f'dPHI{g}_spatial')

    # Find the peak of flux2
    peak_index = 119
    peak_distance = 92.0
    peak_value = flux2_values[peak_index]

    # Add a vertical dashed line at the peak
    ax1.axvline(x=peak_distance, color='g', linestyle='--', linewidth=1.5)
    ax1.annotate(f'AVS Source', 
             xy=(peak_distance, peak_value),
             xytext=(peak_distance + 5, peak_value+0.002),
             arrowprops=dict(arrowstyle='->', color='g'),
             color='g')

    ax1.set_xlabel('Distance to Core Center (cm)')
    ax1.set_ylabel(f'{process_data.capitalize()} dPHI{g}')
    ax1.set_title(fr'{process_data.capitalize()} $\delta \phi_{{{g}}}^{{\text{{pk}}}}$ and $\delta \phi_{{{g}}}^{{\text{{spatial}}}}$ at Centerline (y=0)')
    ax1.set_xlim(unique_distances[2], unique_distances[-2])

    if process_data == 'magnitude':
        ymax = max(max(flux1_values), max(flux2_values))
        ymax *= 1.05
        PHI2g = np.abs(PHI2g)  # Compute magnitude
        ax1.set_ylim(0, ymax)
    elif process_data == 'phase':
        ax1.set_ylim(-180, 180)
    ax1.grid(True)
    ax1.legend(loc='best')

    # Save the figure
    plt.savefig(f'Centerline_y0_{case_name}_{varname}_{process_data}_G{g}.png')

#*************************************************************************************
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', '..', 'INPUTS'))
print("Inputs directory added to sys.path:", inputs_dir)
sys.path.append(inputs_dir)
from OBJECTIVES3_TEST01_2DMG_BIBLIS_CENTER_AVS import *

output_dir = f'OUTPUTS/{case_name}/{case_name}_PK_COMPONENTS/{case_name}_NOISE'

group = 2
# Load data from JSON file
with open(f'{case_name}_NOISE_pk_output.json', 'r') as json_file:
    noise_output = json.load(json_file)

# Access keff and PHI from the loaded data
dPHI1_pk = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI1_pk"]]
dPHI2_pk = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI2_pk"]]
dPHI_pk = dPHI1_pk + dPHI2_pk
dPHI_pk_array = np.array(dPHI_pk)
dPHI_pk_reshaped = dPHI_pk_array.reshape(group, N)

# Load data from JSON file
with open(f'{case_name}_NOISE_spatial_output.json', 'r') as json_file:
    noise_output = json.load(json_file)

# Access keff and PHI from the loaded data
dPHI1_spatial = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI1_spatial"]]
dPHI2_spatial = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI2_spatial"]]
dPHI_spatial = dPHI1_spatial + dPHI2_spatial
dPHI_spatial_array = np.array(dPHI_spatial)
dPHI_spatial_reshaped = dPHI_spatial_array.reshape(group, N)

# Make neighbor indexes
conv = convert_index_2D_rect(D, I_max, J_max)
conv_array = np.array(conv)

##################################################################
for g in range(group):
    plot_1D_centerline_y0_2(dPHI_pk_reshaped[g], dPHI_spatial_reshaped[g], conv, I_max, J_max, dx, dy, g+1, varname='dPHI', process_data='magnitude')
#    plot_1D_centerline_y0_2(dPHI_pk_reshaped[g], dPHI_spatial_reshaped[g], conv, I_max, J_max, dx, dy, g+1, varname='dPHI', process_data='phase')
