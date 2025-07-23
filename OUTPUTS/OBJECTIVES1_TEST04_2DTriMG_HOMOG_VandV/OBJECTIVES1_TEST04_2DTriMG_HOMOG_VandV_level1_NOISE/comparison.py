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

def plot_1D_distance_to_core(PHIg, FLXg, h, I_max, J_max, g, level, varname=None, process_data=None):

    if process_data == 'magnitude':
        PHIg = np.abs(PHIg)  # Compute magnitude
        FLXg = np.abs(FLXg)  # Compute magnitude
    elif process_data == 'phase':
        PHIg = np.degrees(np.angle(PHIg))  # Convert rad to deg
        FLXg = np.degrees(np.angle(FLXg))  # Convert rad to deg
    else:
        pass

    l = 6 * (4 ** (level - 1))
    N_hexx = I_max * J_max * l
    distance_flux_map = defaultdict(list)

    x_center = 0
    y_center = 0
    tolerance = 1e-5  # Define a small tolerance for floating point comparisons

    # Collect all x_base and y_base coordinates to find the centroid
    all_x_coords = []
    all_y_coords = []

    for n in range(N_hexx):
        current_hexx_row = (n // (I_max * l))
        j = n // (I_max * l)
        i = n % (I_max * l)

        if n % 6 == 1 or n % 6 == 2:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h
        elif n % 6 == 3 or n % 6 == 4:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h + h/2
        elif n % 6 == 5 or n % 6 == 0:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h + h

        # Add the x_base and y_base to the lists
        all_x_coords.append(x_base)
        all_y_coords.append(y_base)

    # Compute the centroid
    x_centroid = sum(all_x_coords) / len(all_x_coords)
    y_centroid = sum(all_y_coords) / len(all_y_coords)

    # Track the maximum distance to calculate the radius
    max_distance = 0

    # Now loop through again and translate the coordinates by subtracting the centroid
    for n in range(N_hexx):
        current_hexx_row = (n // (I_max * l))
        j = n // (I_max * l)
        i = n % (I_max * l)

        if n % 6 == 1 or n % 6 == 2:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h
        elif n % 6 == 3 or n % 6 == 4:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h + h/2
        elif n % 6 == 5 or n % 6 == 0:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h + h

        # Translate the coordinates by subtracting the centroid
        x_base_translated = x_base - x_centroid
        y_base_translated = y_base - y_centroid

        # Filter out points where PHIg == 0 and restrict to centerline (y_base_translated near 0)
        if PHIg[n] != 0 and np.abs(y_base_translated) < tolerance:
            signed_distance = x_base_translated
            max_distance = max(max_distance, abs(signed_distance))  # Track the max distance (radius)
            distance_flux_map[signed_distance].append(PHIg[n])

    # Extract maximum flux at each signed distance
    unique_distances = sorted(distance_flux_map.keys())
    flux_values = [max(distance_flux_map[d]) for d in unique_distances]

    # Create an array for analytical flux values corresponding to unique distances
    analytical_flux_values = np.interp(unique_distances, r, FLXg)  # Use linear interpolation to match distances    

    # Initialize empty lists to store distances and flux values within the range [-150, 150]
    filtered_distances = []
    filtered_flux_values = []

#    # Iterate over the unique distances and corresponding flux values
#    for i, d in enumerate(unique_distances):
#        if -150 <= d <= 150:  # Check if the distance is within the range [-150, 150]
#            filtered_distances.append(d)  # Save the distance to the filtered list
#            filtered_flux_values.append(analytical_flux_values[i])  # Save the corresponding flux value

#    # Plot distance vs max flux values
#    plt.figure(figsize=(8, 6))
#    plt.plot(unique_distances, flux_values, 'bo', markersize=5, label='Numerical Flux at Centerline')
#    plt.plot(filtered_distances, filtered_flux_values, 'r-', label='Analytical Flux')
#
#    plt.xlabel('Distance to Core Center')
#    plt.ylabel(f'{process_data} dPHI Values (normalized)')
#    plt.title(f'Group {g} {process_data} dPHI Values vs. Distance to Core Center')
#
#    # Set axis limits from -radius to radius
#    plt.xlim(-150, 150)
#    plt.grid(True)
#    plt.legend()
#    plt.savefig(f'Verification_{varname}_{process_data}_G{g}.png')

    # Calculate relative error
    relative_error = np.abs(np.array(flux_values) - np.array(analytical_flux_values)) / np.array(analytical_flux_values) * 100

    # Plot distance vs max flux values
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot primary y-axis (left)
    ax1.plot(unique_distances, flux_values, 'bo', markersize=5, label='Numerical Flux at Centerline')
    ax1.plot(unique_distances, analytical_flux_values, 'r-', label='Analytical Flux')
    ax1.set_xlabel('Distance to Core Center')
    ax1.set_ylabel(f'{process_data} dPHI Group {g} Values (normalized)')
    ax1.set_title(f'Group {g} {process_data} dPHI Values vs. Distance to Core Center')
    ax1.set_xlim(-150, 150)
    ax1.grid(True)
    ax1.legend(loc='best')

    # Create secondary y-axis (right) for relative error
    ax2 = ax1.twinx()
    ax2.plot(unique_distances, relative_error, 'g--', label='Relative Error (in %)')
    ax2.set_ylabel('Relative Error')
    ax2.legend(loc='best')

    # Save the figure
    plt.savefig(f'Verification_{varname}_{process_data}_G{g}.png')


#    plt.show()

#*************************************************************************************
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES1_TEST04_2DTriMG_HOMOG_VandV import *

#os.chdir('../OUTPUTS/TASK1_TEST16_2DTriMG_HOMOG_VandV/TASK1_TEST16_2DTriMG_HOMOG_VandV_level1_FORWARD')
#sys.path.append(os.getcwd())

output_dir = f'OUTPUTS/{input_name}'

# Load data from JSON file
with open(f'{case_name}_NOISE_output.json', 'r') as json_file:
    noise_output = json.load(json_file)

# Access keff and PHI from the loaded data
dPHI1 = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI1"]]
dPHI2 = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI2"]]
dPHI = dPHI1 + dPHI2
dPHI_array = np.array(dPHI)
dPHI_reshaped = dPHI_array.reshape(group, N_hexx)

# Make neighbor indexes
conv_hexx = convert_hexx(I_max, J_max, D)
conv_tri = convert_tri(I_max, J_max, conv_hexx, level)

#################################################################
# - Analytical solution

## - Cross sections
keff = 1.01241
D1 = 0.5376
D2 = 0.1423
Sigma_a1 = 0.0115
Sigma_a2 = 0.1019
nuSigma_f1 = 0.0057 / 1.01241 *(1 - (1j*omega*Beff)/(1j*omega + l))
nuSigma_f2 = 0.14425 / 1.01241 *(1 - (1j*omega*Beff)/(1j*omega + l))
Sigma_R = 0.0151

Sigma_1 = Sigma_a1 + (1j*omega)/v1 + Sigma_R - nuSigma_f1
Sigma_2 = Sigma_a2 + (1j*omega)/v2

## - Radii
R = 150 #cm
R_ext = R + (2*D1)
r = np.linspace(-R_ext, R_ext, 101)
print(R_ext)

# Equation for mu and la
mu = np.sqrt((-(Sigma_1 * D2 + Sigma_2 * D1) + np.sqrt((Sigma_1 * D2 + Sigma_2 * D1)**2 - 4 * D1 * D2 * (Sigma_1 * Sigma_2 - Sigma_R * nuSigma_f2))) / (2 * D1 * D2))
la = np.sqrt(((Sigma_1 * D2 + Sigma_2 * D1) + np.sqrt((Sigma_1 * D2 + Sigma_2 * D1)**2 - 4 * D1 * D2 * (Sigma_1 * Sigma_2 - Sigma_R * nuSigma_f2))) / (2 * D1 * D2))

c_mu = Sigma_R/(Sigma_2 + D2 * mu**2)
c_la = Sigma_R/(Sigma_2 - D2 * la**2)

# Define dFLX
term1 = 1 / (4 * D2 * (c_la - c_mu))
term2 = (jv(0, mu * np.abs(r)) * yv(0, mu * R) / jv(0, mu * R)) - yv(0, mu * np.abs(r))
term3 = 1 / (2 * np.pi * D2 * (c_la - c_mu))
term4 = kv(0, la * np.abs(r)) - iv(0, la * np.abs(r)) * kv(0, la * R) / iv(0, la * R)

dFLX1 = term1 * 1 * term2 - term3 * 1 * term4
dFLX2 = term1 * c_mu * term2 - term3 * c_la * term4

dFLX = [dFLX1, dFLX2]

#################################################################
for g in range(group):
    plot_1D_distance_to_core(dPHI_reshaped[g], dFLX[g], h, I_max, J_max, g+1, level, varname='dPHI', process_data='magnitude')
    plot_1D_distance_to_core(dPHI_reshaped[g], dFLX[g], h, I_max, J_max, g+1, level, varname='dPHI', process_data='phase')