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

def expand_XS_hexx_2D(group, J_max, I_max, XS, level):
    n = 6 * (4 ** (level - 1))

    XS_temp = np.reshape(XS, (group, I_max, J_max))
    XS_hexx = [[0] * (I_max * J_max * n) for _ in range(group)]

    for g in range(group):
        for j in range(J_max):
            for i in range(I_max):
                m = j * I_max + i
                for k in range(n):
                    XS_hexx[g][m * n + k] = XS_temp[g][j][i]

    return XS_hexx

##############################################################################
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

def plot_triangular(PHIg, x_coords, y_coords, tri_indices, g, cmap='viridis', varname=None, title=None, case_name=None, output_dir=None, solve=None, process_data=None):
    if process_data == 'magnitude':
        PHIg = np.abs(PHIg)  # Compute magnitude
    elif process_data == 'phase':
        PHIg_rad = np.angle(PHIg)  # Compute phase
        PHIg = np.degrees(PHIg_rad)  # Convert rad to deg
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x = []
    y = []

    for i in range(len(x_coords)):
        x.append(x_coords[i]-x_center)
        y.append(y_coords[i]-y_center)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')

    # Create the triangular color plot
    tri_plot = ax.tripcolor(x, y, tri_indices, facecolors=PHIg, cmap=cmap)

    # Add colorbar linked to the tri_plot object
    cbar_label = f'{varname}{g}'
    if solve == 'noise' and process_data == 'phase':
        cbar_label += '_deg'
    elif solve == 'noise' and process_data == 'magnitude':
        cbar_label += '_mag'

    cbar = fig.colorbar(tri_plot, ax=ax, label=cbar_label)
    
    if title:
        plt.title(title)

    # Note: 
    # solve could be:
    # "FORWARD", 
    # "ADJOINT", 
    # "NOISE", "NOISE_GREEN", "NOISE_UNFOLD", "NOISE_dPOWER", 
    # "NOISE_{position_noise}_{type_noise_str}", "NOISE_GREEN_{position_noise}_{type_noise_str}", "NOISE_UNFOLD_{position_noise}_{type_noise_str}", "NOISE_dPOWER_{position_noise}_{type_noise_str}", 
    plt.savefig(f'{case_name}_{solve}_{varname}_{process_data}_G{g}.png')
    plt.close(fig)

#################################################################
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES1_TEST05_2DTriMG_VVER400_VandV import *

output_dir = f'OUTPUTS/{input_name}'

# Make neighbor indexes
conv_hexx = convert_hexx(I_max, J_max, D)
conv_tri, conv_hexx_ext = convert_tri(I_max, J_max, conv_hexx, level)
conv_tri_array = np.array(conv_tri)
conv_neighbor, tri_indices, x, y, all_triangles = calculate_neighbors_2D(s, I_max, J_max, conv_hexx, level)

#################################################################
FLX_hexx = expand_XS_hexx_2D(group, J_max, I_max, FLX, level)
FLX_temp = np.zeros((group, max(conv_tri)))
for g in range(group):
    for m in range(len(conv_tri)):
        if conv_tri[m] != 0:
            FLX_temp[g][conv_tri[m]-1] = FLX_hexx[g][m]

max_FLX = np.max(FLX_temp)
FLX_temp = FLX_temp / max_FLX

##################################################################
# Load data from JSON file
with open(f'{case_name}_FORWARD_output.json', 'r') as json_file:
    forward_output = json.load(json_file)

# Access keff and PHI from the loaded data
keff = forward_output["keff"]
PHI1 = forward_output["PHI1_FORWARD"]
PHI2 = forward_output["PHI2_FORWARD"]
PHI = PHI1 + PHI2
PHI = np.reshape(PHI, (group, N_hexx))

POWER = np.zeros_like(PHI)
for g in range(group):
    for n in range(len(PHI1)):
        POWER[g][n] = PHI[g][n] * 1.0

print(J_max, I_max)
n = 6 * (4 ** (level - 1))
POWER_asmb = np.zeros_like(FLX)
for g in range(group):
    for j in range(J_max):
        for i in range(I_max):
            m = j * I_max + i
            block = POWER[g][m*n:(m+1)*n]
            POWER_asmb[g][m] = np.sum(block)

dV_asmb = 1 #h**2/4*np.sqrt(3)

PHI_norm_reshaped = POWER_asmb / dV_asmb
PHI_norm_hexx = expand_XS_hexx_2D(group, J_max, I_max, PHI_norm_reshaped, level)

PHI_norm_temp = np.zeros((group, max(conv_tri)))
for g in range(group):
    for m in range(len(conv_tri)):
        if conv_tri[m] != 0:
            PHI_norm_temp[g][conv_tri[m]-1] = PHI_norm_hexx[g][m]

max_PHI = np.max(PHI_norm_temp)
PHI_norm_temp = PHI_norm_temp / max_PHI

diff_PHI_temp = np.zeros((group, max(conv_tri)))
for g in range(group):
    for m in range(max(conv_tri)):
        if conv_tri[m] != 0:
            diff_PHI_temp[g][m] = np.abs((FLX_temp[g][m] - PHI_norm_temp[g][m]) / FLX_temp[g][m]) * 100

##################################################################
for g in range(group):
    plot_triangular(FLX_temp[g], x, y, tri_indices, g+1, cmap='viridis', varname='FLX', title=f'2D Plot of FLX{g+1} Hexx Magnitude', case_name=case_name, output_dir=output_dir, solve='FORWARD', process_data="magnitude")
    plot_triangular(PHI_norm_temp[g], x, y, tri_indices, g+1, cmap='viridis', varname='PHI_normalized', title=f'2D Plot of PHI{g+1}_normalized Hexx Magnitude', case_name=case_name, output_dir=output_dir, solve='FORWARD', process_data="magnitude")
    plot_triangular(diff_PHI_temp[g], x, y, tri_indices, g+1, cmap='viridis', varname='diff_PHI', title=f'2D Plot of diff_PHI{g+1} Hexx Magnitude', case_name=case_name, output_dir=output_dir, solve='FORWARD', process_data="magnitude")
