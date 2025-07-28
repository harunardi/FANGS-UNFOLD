import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import json
import os
import sys

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

def plot_triangular_level2(PHIg, h, I_max, J_max, g, level, cmap='viridis', varname=None, title=None, case_name=None, output_dir=None, solve=None, process_data=None, noise_pos=0, type_noise=0):
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
        if n % l == 1:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3))
            y_base = j * 3 * h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 2:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3))
            y_base = j * 3 * h
            values_right.append(PHIg[n])
            right_vertice = [(x_base + h * np.sqrt(3) / 3 + h / np.sqrt(3), y_base), 
                             (x_base + h * np.sqrt(3) / 3 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base + h * np.sqrt(3) / 3 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice)
        elif n % l == 3:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) - h * np.sqrt(3)/2
            y_base = j * 3 * h + h/2
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 4:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3))
            y_base = j * 3 * h + h/2
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice) 
        elif n % l == 5:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) + h * np.sqrt(3)/2
            y_base = j * 3 * h + h/2
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 6:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) + h * np.sqrt(3)
            y_base = j * 3 * h + h/2
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice)
        elif n % l == 7:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) - h * np.sqrt(3)/2
            y_base = j * 3 * h + h
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice)
        elif n % l == 8:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3))
            y_base = j * 3 * h + h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 9:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) + h * np.sqrt(3)/2
            y_base = j * 3 * h + h
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice)
        elif n % l == 10:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) + h * np.sqrt(3)
            y_base = j * 3 * h + h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 11:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) - h * np.sqrt(3)/2
            y_base = j * 3 * h + (3/2) * h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 12:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3))
            y_base = j * 3 * h + (3/2) * h
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice) 
        elif n % l == 13:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) + h * np.sqrt(3)/2
            y_base = j * 3 * h + (3/2) * h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 14:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) + h * np.sqrt(3)
            y_base = j * 3 * h + (3/2) * h
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice)
        elif n % l == 15:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) - h * np.sqrt(3)/2
            y_base = j * 3 * h + 2 * h
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice)
        elif n % l == 16:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3))
            y_base = j * 3 * h + 2 * h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 17:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) + h * np.sqrt(3)/2
            y_base = j * 3 * h + 2 * h
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice)
        elif n % l == 18:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) + h * np.sqrt(3)
            y_base = j * 3 * h + 2 * h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 19:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) - h * np.sqrt(3)/2
            y_base = j * 3 * h + (5/2) * h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 20:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3))
            y_base = j * 3 * h + (5/2) * h
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice) 
        elif n % l == 21:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) + h * np.sqrt(3)/2
            y_base = j * 3 * h + (5/2) * h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 22:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3)) + h * np.sqrt(3)
            y_base = j * 3 * h + (5/2) * h
            values_right.append(PHIg[n])
            right_vertice = [(x_base - h * np.sqrt(3) / 6 + h / np.sqrt(3), y_base), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base - h/2), 
                             (x_base - h * np.sqrt(3) / 6 - h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_right.append(right_vertice)
        elif n % l == 23:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3))
            y_base = j * 3 * h + 3 * h
            values_left.append(PHIg[n])
            left_vertice = [(x_base - h / np.sqrt(3), y_base), 
                            (x_base + h / (np.sqrt(3)*2), y_base - h/2), 
                            (x_base + h / (np.sqrt(3)*2), y_base + h/2)]
            vertices_left.append(left_vertice)
        elif n % l == 0:
            x_base = (i // l) * 2 * h * np.sqrt(3) + (current_hexx_row * h * np.sqrt(3))
            y_base = j * 3 * h + 3 * h
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
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
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
#    if solve == "forward":
#        plt.savefig(f'{output_dir}/{case_name}_FORWARD/{case_name}_FORWARD_{varname}_G{g}.png')
#    elif solve == "adjoint":
#        plt.savefig(f'{output_dir}/{case_name}_ADJOINT/{case_name}_ADJOINT_{varname}_G{g}.png')
#    elif solve == "noise":
#        if noise_pos == 1:
#            position_noise = 'CENTER'
#            if type_noise == 1:
#                type_noise_str = 'AVS'
#                plt.savefig(f'{output_dir}/{case_name}_NOISE_{position_noise}_{type_noise_str}/{case_name}_NOISE_{varname}_{process_data}_G{g}.png')
#        elif noise_pos == 2:
#            position_noise = 'NONCENTER'
#            if type_noise == 1:
#                type_noise_str = 'AVS'
#                plt.savefig(f'{output_dir}/{case_name}_NOISE_{position_noise}_{type_noise_str}/{case_name}_NOISE_{varname}_{process_data}_G{g}.png')
#        else:
#            plt.savefig(f'{output_dir}/{case_name}_NOISE/{case_name}_NOISE_{varname}_{process_data}_G{g}.png')
#    elif solve == "noise_green":
#        if noise_pos == 1:
#            position_noise = 'CENTER'
#            if type_noise == 1:
#                type_noise_str = 'AVS'
#                plt.savefig(f'{output_dir}/{case_name}_NOISE_GREEN_{position_noise}_{type_noise_str}/{case_name}_NOISE_GREEN_{varname}_{process_data}_G{g}.png')
#        elif noise_pos == 2:
#            position_noise = 'NONCENTER'
#            if type_noise == 1:
#                type_noise_str = 'AVS'
#                plt.savefig(f'{output_dir}/{case_name}_NOISE_GREEN_{position_noise}_{type_noise_str}/{case_name}_NOISE_GREEN_{varname}_{process_data}_G{g}.png')
#        else:
#            plt.savefig(f'{output_dir}/{case_name}_NOISE_GREEN/{case_name}_NOISE_GREEN_{varname}_{process_data}_G{g}.png')
#    elif solve == "noise_unfold":
#        if noise_pos == 1:
#            position_noise = 'CENTER'
#            if type_noise == 1:
#                type_noise_str = 'AVS'
#                plt.savefig(f'{output_dir}/{case_name}_NOISE_GREEN_{position_noise}_{type_noise_str}/{case_name}_NOISE_UNFOLD_{varname}_{process_data}_G{g}.png')
#        elif noise_pos == 2:
#            position_noise = 'NONCENTER'
#            if type_noise == 1:
#                type_noise_str = 'AVS'
#                plt.savefig(f'{output_dir}/{case_name}_NOISE_GREEN_{position_noise}_{type_noise_str}/{case_name}_NOISE_UNFOLD_{varname}_{process_data}_G{g}.png')
#        else:
#            plt.savefig(f'{output_dir}/{case_name}_NOISE_GREEN/{case_name}_NOISE_UNFOLD_{varname}_{process_data}_G{g}.png')
#    elif solve == "power_perturbation":
#        plt.savefig(f'{output_dir}/{case_name}_POWER_PERTURBATION/{case_name}_dPOWER_{varname}_{process_data}_G{g}.png')

    plt.savefig(f'{case_name}_FORWARD_{varname}_G{g}.png')
    plt.close(fig)
