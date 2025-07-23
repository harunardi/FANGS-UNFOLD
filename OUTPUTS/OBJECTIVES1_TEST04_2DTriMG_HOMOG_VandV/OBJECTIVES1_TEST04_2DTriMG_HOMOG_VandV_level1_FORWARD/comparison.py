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

def plot_1D_distance_to_core(PHIg, FLXg, h, I_max, J_max, g, level, varname=None, process_data=None):
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

#    # Plot distance vs max flux values
#    plt.figure(figsize=(8, 6))
#    plt.plot(unique_distances, flux_values, 'bo', markersize=5, label='Numerical Flux at Centerline')
#
#    # Plot analytical flux values
#    plt.plot(unique_distances, analytical_flux_values, 'r-', label='Analytical Flux')  # Add this line
#
#    plt.xlabel('Distance to Core Center')
#    plt.ylabel('Forward Flux Values (normalized)')
#    plt.title('Forward Flux Values vs. Distance to Core Center')
#
#    # Set axis limits from -radius to radius
#    plt.xlim(-max_distance, max_distance)
#    plt.grid(True)
#    plt.legend()
#    plt.savefig(f'Verification_{varname}_{process_data}_G{g}.png')

    # Calculate relative error
    relative_error = np.abs(np.array(flux_values) - np.array(analytical_flux_values)) / np.array(analytical_flux_values)

    # Plot distance vs max flux values
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot primary y-axis (left)
    ax1.plot(unique_distances, flux_values, 'bo', markersize=5, label='Numerical Flux at Centerline')
    ax1.plot(unique_distances, analytical_flux_values, 'r-', label='Analytical Flux')
    ax1.set_xlabel('Distance to Core Center')
    ax1.set_ylabel(f'Forward Flux Group {g+1} Values (normalized)')
    ax1.set_title(f'Forward Flux Group {g+1} Values vs. Distance to Core Center')
    ax1.set_xlim(-max_distance, max_distance)
    ax1.grid(True)
    ax1.legend()

    # Create secondary y-axis (right) for relative error
    ax2 = ax1.twinx()
    ax2.plot(unique_distances, relative_error, 'g--', label='Relative Error')
    ax2.set_ylabel('Relative Error')
    ax2.legend(loc='upper right')

    # Save the figure
    plt.savefig(f'Verification_{varname}_{process_data}_G{g}.png')
#    plt.show()

#################################################################
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES1_TEST04_2DTriMG_HOMOG_VandV import *

#os.chdir('../OUTPUTS/TASK1_TEST16_2DTriMG_HOMOG_VandV/TASK1_TEST16_2DTriMG_HOMOG_VandV_level1_FORWARD')
#sys.path.append(os.getcwd())

output_dir = f'OUTPUTS/{input_name}'

# Load data from JSON file
with open(f'{case_name}_FORWARD_output.json', 'r') as json_file:
    forward_output = json.load(json_file)

# Access keff and PHI from the loaded data
keff = forward_output["keff"]
PHI1 = forward_output["PHI1_FORWARD"]
PHI2 = forward_output["PHI2_FORWARD"]
PHI = PHI1 + PHI2
PHI_array = np.array(PHI)
PHI_reshaped = PHI_array.reshape(group, N_hexx)

# Make neighbor indexes
conv_hexx = convert_hexx(I_max, J_max, D)
conv_tri = convert_tri(I_max, J_max, conv_hexx, level)

#################################################################
# - Analytical solution

## - Cross sections
D1 = 0.5376
D2 = 0.1423
Sigma_a1 = 0.0115
Sigma_a2 = 0.1019
nuSigma_f1 = 0.0057
nuSigma_f2 = 0.14425
Sigma_R = 0.0151

## - Radii
R = 150 #cm
j_0 = sp.jn_zeros(0, 1)[0]
R_ext = R + (2*D1)
r = np.linspace(-R_ext, R_ext, 101)
B_g = j_0/R_ext

c_mu = Sigma_R/(Sigma_a2 + (D2 * B_g**2))

J0_values = j0(B_g * r)
FLX1 = 1 * J0_values
FLX2 = c_mu * J0_values

FLX = [FLX1, FLX2]

#################################################################
for g in range(group):
    plot_1D_distance_to_core(PHI_reshaped[g], FLX[g], h, I_max, J_max, g+1, level, varname='PHI', process_data='magnitude')
