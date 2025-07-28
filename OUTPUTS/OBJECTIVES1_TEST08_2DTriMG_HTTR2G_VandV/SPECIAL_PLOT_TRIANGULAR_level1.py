import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from collections import defaultdict
import json
import os
import sys

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

def plot_triangular_level1(PHIg, h, I_max, J_max, g, level, cmap='viridis', varname=None, title=None, case_name=None, output_dir=None, solve=None, process_data=None, noise_pos=0, type_noise=0):
    l = 6 * (4 ** (level - 1))
    N_hexx = I_max * J_max * l

    if solve == 'noise' or solve == 'noise_green' or solve == 'noise_unfold':
        if process_data == 'magnitude':
            PHIg = np.abs(PHIg)  # Compute magnitude
        elif process_data == 'phase':
            PHIg_rad = np.angle(PHIg)  # Compute phase
            PHIg = np.degrees(PHIg_rad)  # Convert rad to deg
    else:
        pass

    values_right, values_left = [], []
    vertices_left, vertices_right = [], []

    for n in range(N_hexx):
        current_hexx_row = (n // (I_max * l))
        j = n // (I_max * l)
        i = n % (I_max * l)
        if n % 6 == 1:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % 6 == 2:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h
            values_right.append(PHIg[n])
            right_vertice = [(x_base + h * np.sqrt(3) / 3 + h / np.sqrt(3), y_base), 
                             (x_base + h * np.sqrt(3) / 3 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base + h * np.sqrt(3) / 3 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice)
        elif n % 6 == 3:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h + h/2
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice) 
        elif n % 6 == 4:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h + h/2 
            values_left.append(PHIg[n])
            left_vertice = [(x_base + h * np.sqrt(3) / 2 - h / np.sqrt(3), y_base), 
                            (x_base + h * np.sqrt(3) / 2 + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h * np.sqrt(3) / 2 + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % 6 == 5:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h + h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % 6 == 0:
            x_base = (i // 6) * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)/2)
            y_base = j * (3/2) * h + h
            values_right.append(PHIg[n])
            right_vertice = [(x_base + h * np.sqrt(3) / 3 + h / np.sqrt(3), y_base), 
                             (x_base + h * np.sqrt(3) / 3 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base + h * np.sqrt(3) / 3 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice)

    # Filter out vertices where corresponding values are 0
    filtered_vertices_left = [v for v, val in zip(vertices_left, values_left) if not np.isnan(val) and val != 0]
    filtered_values_left = [val for val in values_left if not np.isnan(val) and val != 0]
    filtered_vertices_right = [v for v, val in zip(vertices_right, values_right) if not np.isnan(val) and val != 0]
    filtered_values_right = [val for val in values_right if not np.isnan(val) and val != 0]

    # Combine left and right filtered vertices and values
    all_filtered_vertices = filtered_vertices_left + filtered_vertices_right
    all_filtered_values = filtered_values_left + filtered_values_right

    # Calculate limits for x and y based on vertices
    x_coords = [x for verts in all_filtered_vertices for x, y in verts]
    y_coords = [y for verts in all_filtered_vertices for x, y in verts]
    x_min, x_max = np.nanmin(x_coords), np.nanmax(x_coords)
    y_min, y_max = np.nanmin(y_coords), np.nanmax(y_coords)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Translate vertices to center at (0, 0)
    translated_vertices = []
    for verts in all_filtered_vertices:
        translated_verts = [(x - x_center, y - y_center) for x, y in verts]
        translated_vertices.append(translated_verts)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Create polygon patches for left and right triangles
    patches = [Polygon(vertices, closed=True) for vertices in translated_vertices]
    values = np.array(all_filtered_values)

    # Create PatchCollection with a single colormap
    p = PatchCollection(patches, cmap=cmap)
    p.set_array(values)

    # Add collection to plot
    ax.add_collection(p)

    # Set limits
    ax.set_xlim(x_min - x_center, x_max - x_center)
    ax.set_ylim(y_min - y_center, y_max - y_center)
    print(x_center, x_min, x_max)

    # Add colorbar
    if solve == 'noise':
        if process_data == 'phase':
            plt.colorbar(p, ax=ax, label=f'{varname}{g}_deg')
        else:
            plt.colorbar(p, ax=ax, label=f'{varname}{g}_mag')
    else:
        plt.colorbar(p, ax=ax, label=f'{varname}{g}')

    if title:
        plt.title(title)

    plt.savefig(f'{case_name}_FORWARD_{varname}_G{g}.png')
    plt.close(fig)

def plot_1D_distance_to_core(PHIg, h, I_max, J_max, g, level, varname=None, case_name=None, output_dir=None, solve=None):
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

    if solve == "forward":
        np.savetxt(f"{output_dir}/{case_name}_FORWARD/plot_unique_distances.txt", unique_distances)
        np.savetxt(f"{output_dir}/{case_name}_FORWARD/plot_{varname}_centerline_g{g}.txt", flux_values)
    elif solve == "adjoint":
        np.savetxt(f"{output_dir}/{case_name}_ADJOINT/plot_unique_distances.txt", unique_distances)
        np.savetxt(f"{output_dir}/{case_name}_ADJOINT/plot_{varname}_centerline_g{g}.txt", flux_values)
    elif solve == "noise":
        np.savetxt(f"{output_dir}/{case_name}_NOISE/plot_unique_distances.txt", unique_distances)
        np.savetxt(f"{output_dir}/{case_name}_NOISE/plot_{varname}_centerline_g{g}.txt", flux_values)

    # Plot distance vs max flux values
    plt.figure(figsize=(8, 6))
    plt.plot(unique_distances, flux_values, 'bo', markersize=5, label='Max Flux at Distance (Centerline)')
    plt.xlabel('Distance to Core Center (After Translation)')
    plt.ylabel('Max Flux Values')
    plt.title('Max Flux Values vs. Distance to Core Center (Centerline, Translated)')

    # Set axis limits from -radius to radius
    plt.xlim(-max_distance, max_distance)
    plt.grid(True)
    plt.legend()
    plt.show()

