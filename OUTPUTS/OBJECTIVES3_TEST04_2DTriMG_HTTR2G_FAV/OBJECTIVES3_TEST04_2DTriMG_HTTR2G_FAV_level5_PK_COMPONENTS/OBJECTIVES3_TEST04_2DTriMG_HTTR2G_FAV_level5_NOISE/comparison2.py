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

def plot_1D_centerline_y0(PHI1g, PHI2g, h, I_max, J_max, g, level, varname=None, process_data=None):

    if process_data == 'magnitude':
        PHI1g = np.abs(PHI1g)  # Compute magnitude
        PHI2g = np.abs(PHI2g)  # Compute magnitude
    elif process_data == 'phase':
        PHI1g = np.degrees(np.angle(PHI1g))  # Convert rad to deg
        PHI2g = np.degrees(np.angle(PHI2g))  # Convert rad to deg
    else:
        pass

    l = 6 * (4 ** (level - 1))
    N_hexx = I_max * J_max * l
    distance_flux1_map = defaultdict(list)
    distance_flux2_map = defaultdict(list)

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

        # Filter out points where PHI1g == 0 and restrict to centerline (y_base_translated near 0)
        if PHI1g[n] != 0 and np.abs(y_base_translated) < tolerance:
            signed_distance = x_base_translated
            max_distance = max(max_distance, abs(signed_distance))  # Track the max distance (radius)
            distance_flux1_map[signed_distance].append(PHI1g[n])

        if PHI2g[n] != 0 and np.abs(y_base_translated) < tolerance:
            signed_distance = x_base_translated
            max_distance = max(max_distance, abs(signed_distance))  # Track the max distance (radius)
            distance_flux2_map[signed_distance].append(PHI2g[n])

    # Extract maximum flux at each signed distance
    unique_distances = sorted(distance_flux1_map.keys())
    flux1_values = [max(distance_flux1_map[d]) for d in unique_distances]
    flux2_values = [max(distance_flux2_map[d]) for d in unique_distances]

    # Initialize empty lists to store distances and flux values within the range [-150, 150]
    filtered_distances = []
    filtered_flux1_values = []
    filtered_flux2_values = []

    # Plot distance vs max flux values
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot primary y-axis (left)
    ax1.plot(unique_distances, flux1_values, 'b', markersize=5, label='dPHI_pk at Centerline')
    ax1.plot(unique_distances, flux2_values, 'r', markersize=5, label='dPHI_spatial at Centerline')
    ax1.set_xlabel('Distance to Core Center')
    ax1.set_ylabel(f'{process_data} dPHI Group {g} Values (normalized)')
    ax1.set_title(f'Group {g} {process_data} dPHI Values vs. Distance to Core Center')
    ax1.set_xlim(unique_distances[0], unique_distances[-1])
    ax1.grid(True)
    ax1.legend(loc='best')

    # Save the figure
    plt.savefig(f'Centerline_y0_{varname}_{process_data}_G{g}.png')

def plot_1D_centerline_y0_2(PHI1g, PHI2g, x_coords, y_coords, conv_tri, I_max, J_max, g, level, varname=None, process_data=None):

    if process_data == 'magnitude':
        PHI1g = np.abs(PHI1g)  # Compute magnitude
        PHI2g = np.abs(PHI2g)  # Compute magnitude
    elif process_data == 'phase':
        PHI1g = np.degrees(np.angle(PHI1g))  # Convert rad to deg
        PHI2g = np.degrees(np.angle(PHI2g))  # Convert rad to deg
    else:
        pass

    conv_tri_array = np.array(conv_tri)
    PHI1g_temp = np.zeros(max(conv_tri) * group)
    PHI2g_temp = np.zeros(max(conv_tri) * group)
    for g1 in range(group):
        PHI1_indices = g1 * max(conv_tri) + (conv_tri_array - 1)
        PHI1g_temp[PHI1_indices] = PHI1g
        PHI2g_temp[PHI1_indices] = PHI2g

    l = 6 * (4 ** (level - 1))
    N_hexx = I_max * J_max * l
    distance_flux1_map = defaultdict(list)
    distance_flux2_map = defaultdict(list)

    tolerance = 1e-5  # Define a small tolerance for floating point comparisons

    # Recentering the coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x = []
    y = []

    for i in range(len(x_coords)):
        x.append(x_coords[i]-x_center)
        y.append(y_coords[i]-y_center)

    centroids = []
    for tri in tri_indices:
        x_c = (x[tri[0]] + x[tri[1]] + x[tri[2]]) / 3
        y_c = (y[tri[0]] + y[tri[1]] + y[tri[2]]) / 3
        centroids.append((x_c, y_c))

    max_distance = 0
    print(len(centroids), N_hexx)
    for n, (x_c, y_c) in enumerate(centroids):
        if PHI1g_temp[n] != 0 and abs(y_c) < tolerance:
            signed_distance = x_c
            max_distance = max(max_distance, abs(signed_distance))
            distance_flux1_map[signed_distance].append(PHI1g_temp[n])
        if PHI2g_temp[n] != 0 and abs(y_c) < tolerance:
            signed_distance = x_c
            max_distance = max(max_distance, abs(signed_distance))
            distance_flux2_map[signed_distance].append(PHI2g_temp[n])

    # Extract maximum flux at each signed distance
    unique_distances = sorted(distance_flux1_map.keys())
    flux1_values = [max(distance_flux1_map[d]) for d in unique_distances]
    flux2_values = [max(distance_flux2_map[d]) for d in unique_distances]

    # Plot distance vs max flux values
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot primary y-axis (left)
    ax1.plot(unique_distances, flux1_values, 'b', markersize=5, label=f'dPHI{g}_pk at Centerline')
    ax1.plot(unique_distances, flux2_values, 'r', markersize=5, label=f'dPHI{g}_spatial at Centerline')

    # Find the peak of flux2
    peak_index1 = 224
    print(peak_index1, unique_distances[peak_index1], flux2_values[peak_index1])
    peak_distance1 = unique_distances[peak_index1]
    peak_value1 = flux2_values[peak_index1]

    # Add a vertical dashed line at the peak
    ax1.axvline(x=peak_distance1, color='g', linestyle='--', linewidth=1.5)
    ax1.annotate(f'FAV1 Source', 
             xy=(peak_distance1, peak_value1),
             xytext=(peak_distance1 + 5, peak_value1 + 0.002),
             arrowprops=dict(arrowstyle='->', color='g'),
             color='g')

    peak_index2 = 243
    print(peak_index2, unique_distances[peak_index2], flux2_values[peak_index2])
    peak_distance2 = unique_distances[peak_index2]
    peak_value2 = flux2_values[peak_index2]

    # Add a vertical dashed line at the peak
    ax1.axvline(x=peak_distance2, color='g', linestyle='--', linewidth=1.5)
    ax1.annotate(f'FAV2 Source', 
             xy=(peak_distance2, peak_value2),
             xytext=(peak_distance2 + 5, peak_value2 + 0.002),
             arrowprops=dict(arrowstyle='->', color='g'),
             color='g')
    
    ax1.set_xlabel('Distance to Core Center')
    ax1.set_ylabel(f'{process_data} dPHI{g}')
    ax1.set_title(f'Group {g} {process_data} dPHI Values vs. Distance to Core Center')
    ax1.set_xlim(unique_distances[0], unique_distances[-1])
    ax1.grid(True)
    ax1.legend(loc='best')

    # Save the figure
    plt.savefig(f'Centerline_y0_{case_name}_{varname}_{process_data}_G{g}.png')

#*************************************************************************************
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES3_TEST04_2DTriMG_HTTR2G_FAV import *

output_dir = f'OUTPUTS/{input_name}/{case_name}_PK_COMPONENTS/{case_name}_NOISE'

# Load data from JSON file
with open(f'{case_name}_NOISE_pk_output.json', 'r') as json_file:
    noise_output = json.load(json_file)

# Access keff and PHI from the loaded data
dPHI1_pk = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI1_pk"]]
dPHI2_pk = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI2_pk"]]
dPHI_pk = dPHI1_pk + dPHI2_pk
dPHI_pk_array = np.array(dPHI_pk)
dPHI_pk_reshaped = dPHI_pk_array.reshape(group, N_hexx)

# Load data from JSON file
with open(f'{case_name}_NOISE_spatial_output.json', 'r') as json_file:
    noise_output = json.load(json_file)

# Access keff and PHI from the loaded data
dPHI1_spatial = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI1_spatial"]]
dPHI2_spatial = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI2_spatial"]]
dPHI_spatial = dPHI1_spatial + dPHI2_spatial
dPHI_spatial_array = np.array(dPHI_spatial)
dPHI_spatial_reshaped = dPHI_spatial_array.reshape(group, N_hexx)

# Make neighbor indexes
conv_hexx = convert_2D_hexx(I_max, J_max, D)
conv_tri, conv_hexx_ext = convert_2D_tri(I_max, J_max, conv_hexx, level)
conv_tri_array = np.array(conv_tri)
conv_neighbor, tri_indices, x, y, all_triangles = calculate_neighbors_2D(s, I_max, J_max, conv_hexx, level)

##################################################################
#for g in range(group):
#    plot_1D_centerline_y0(dPHI_pk_reshaped[g], dPHI_spatial_reshaped[g], h, I_max, J_max, g+1, level, varname='dPHI', process_data='magnitude')
#    plot_1D_centerline_y0(dPHI_pk_reshaped[g], dPHI_spatial_reshaped[g], h, I_max, J_max, g+1, level, varname='dPHI', process_data='phase')

for g in range(group):
    plot_1D_centerline_y0_2(dPHI_pk_reshaped[g], dPHI_spatial_reshaped[g], x, y, conv_tri, I_max, J_max, g+1, level, varname='dPHI', process_data='magnitude')
    plot_1D_centerline_y0_2(dPHI_pk_reshaped[g], dPHI_spatial_reshaped[g], x, y, conv_tri, I_max, J_max, g+1, level, varname='dPHI', process_data='phase')
