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
def convert_2D_hexx(I_max, J_max, D):
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
def convert_2D_tri(I_max, J_max, conv_hexx, level):
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

def generate_pointy_hex_grid(flat_to_flat_distance, I_max, J_max):
    """
    Generate a pointy hexagonal grid using the flat-to-flat distance.
    Parameters:
        flat_to_flat_distance : float
            Flat-to-flat distance of the hexagon.
        I_max, J_max : int
            Number of hexagons along x and y axes.
    Returns:
        hex_centers : list of tuples
            Centers of the hexagons.
        vertices : list of tuples
            Vertices of the hexagon.
    """
    # Calculate radius from flat-to-flat distance
    radius = flat_to_flat_distance / np.sqrt(3)

    # Hexagon vertices (rotated by 30 degrees for pointy-topped)
    hex_vertices = [
        (radius * np.cos(np.pi/6 + 2 * np.pi * k / 6), 
         radius * np.sin(np.pi/6 + 2 * np.pi * k / 6))
        for k in range(6)
    ]

    # Hexagon centers
    hex_centers = []
    for j in range(J_max):
        for i in range(I_max):
            x_offset = radius * np.sqrt(3) * i + radius * np.sqrt(3) / 2 * j
            y_offset = radius * 1.5 * j
            hex_centers.append((x_offset, y_offset))

    return hex_centers, hex_vertices

def subdivide_triangle(p1, p2, p3, level):
    """
    Recursively subdivide a triangle into smaller triangles.
    """
    if level == 1:
        return [(p1, p2, p3)]
    
    mid1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    mid2 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
    mid3 = ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2)
    
    return (
        subdivide_triangle(p1, mid1, mid3, level - 1) +
        subdivide_triangle(mid1, p2, mid2, level - 1) +
        subdivide_triangle(mid3, mid2, p3, level - 1) +
        subdivide_triangle(mid1, mid2, mid3, level - 1)
    )

def subdivide_pointy_hexagon(center, vertices, level):
    """
    Subdivide a pointy hexagon into smaller triangles.
    """
    triangles = []
    for i in range(len(vertices)):
        p1 = center
        p2 = vertices[i]
        p3 = vertices[(i + 1) % len(vertices)]
        triangles += subdivide_triangle(p1, p2, p3, level)
    return triangles

def round_vertex(vertex, precision=6):
    """
    Round vertex coordinates to a fixed precision.
    """
    return tuple(round(coord, precision) for coord in vertex)

def find_triangle_neighbors_2D(triangles, precision=6):
    """
    Find neighbors for each triangle globally based on shared edges.
    Assign -1 for neighbors on the boundary.
    Each triangle will have exactly 3 neighbors.
    """
    edge_map = {}
    neighbors = {i: [-1, -1, -1] for i in range(len(triangles))}  # Initialize with -1 for boundaries

    # Step 1: Map edges to triangles
    for tri_idx, vertices in enumerate(triangles):
        vertices = [round_vertex(v, precision) for v in vertices]
        edges = [
            tuple(sorted((vertices[0], vertices[1]))),
            tuple(sorted((vertices[1], vertices[2]))),
            tuple(sorted((vertices[2], vertices[0]))),
        ]
        for edge in edges:
            if edge in edge_map:
                # Shared edge found
                neighbor_idx = edge_map[edge]
                # Assign neighbors for both triangles
                for i in range(3):
                    if neighbors[tri_idx][i] == -1:
                        neighbors[tri_idx][i] = neighbor_idx
                        break
                for i in range(3):
                    if neighbors[neighbor_idx][i] == -1:
                        neighbors[neighbor_idx][i] = tri_idx
                        break
            else:
                # Map the edge to the current triangle
                edge_map[edge] = tri_idx

    return neighbors

def calculate_neighbors_2D(s, I_max, J_max, conv_hexx, level):
    """
    Do all the necessary calculations to get triangle neighbors.
    """
    # Generate grid
    hex_centers, hex_vertices = generate_pointy_hex_grid(s, I_max, J_max)

    # Subdivide hexagons
    all_triangles = []
    for i, center in enumerate(hex_centers):
        if conv_hexx[i] != 0:
            shifted_vertices = [(vx + center[0], vy + center[1]) for vx, vy in hex_vertices]
            all_triangles += subdivide_pointy_hexagon(center, shifted_vertices, level)

    # Find neighbors with debugging
    triangle_neighbors_global = find_triangle_neighbors_2D(all_triangles, precision=6)

    conv_neighbor = []
    for idx, neighbors in triangle_neighbors_global.items():
        conv_neighbor.append(neighbors)

    # Extract triangle coordinates for plotting
    x = [v[0] for triangle in all_triangles for v in triangle]
    y = [v[1] for triangle in all_triangles for v in triangle]
    tri_indices = np.arange(len(x)).reshape(-1, 3)

    return conv_neighbor, tri_indices, x, y, all_triangles

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
    print(f"Computed centroid: ({x_centroid}, {y_centroid})")

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

def plot_1D_centerline_y0(PHIg, h, I_max, J_max, g, level, varname=None, process_data=None):

    if process_data == 'magnitude':
        PHIg = np.abs(PHIg)  # Compute magnitude
    elif process_data == 'phase':
        PHIg = np.degrees(np.angle(PHIg))  # Convert rad to deg
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
    print(f"Computed centroid: ({x_centroid}, {y_centroid})")

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

    # Initialize empty lists to store distances and flux values within the range [-150, 150]
    filtered_distances = []
    filtered_flux_values = []

    # Plot distance vs max flux values
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot primary y-axis (left)
    ax1.plot(unique_distances, flux_values, 'b', markersize=5, label='Numerical Flux at Centerline')
    ax1.set_xlabel('Distance to Core Center')
    ax1.set_ylabel(f'{process_data} dPHI Group {g} Values (normalized)')
    ax1.set_title(f'Group {g} {process_data} dPHI Values vs. Distance to Core Center')
    ax1.set_xlim(-150, 150)
    ax1.grid(True)
    ax1.legend(loc='best')

    # Save the figure
    plt.savefig(f'Centerline_y0_{varname}_{process_data}_G{g}.png')

#*************************************************************************************
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES1_TEST04_2DTriMG_HOMOG_VandV import *

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
conv_hexx = convert_2D_hexx(I_max, J_max, D)
conv_tri, conv_hexx_ext = convert_2D_tri(I_max, J_max, conv_hexx, level)
conv_tri_array = np.array(conv_tri)
conv_neighbor, tri_indices, x, y, all_triangles = calculate_neighbors_2D(s, I_max, J_max, conv_hexx, level)

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
    plot_1D_centerline_y0(dPHI_reshaped[g], h, I_max, J_max, g+1, level, varname='dPHI', process_data='magnitude')
    plot_1D_centerline_y0(dPHI_reshaped[g], h, I_max, J_max, g+1, level, varname='dPHI', process_data='phase')