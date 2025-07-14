import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

#######################################################################################################
def plot_1D_power(solver_type, data, x, g, varname=None, case_name=None, title=None):
    plt.clf()
    plt.figure()
    plt.plot(x, data, 'g-', label=f'Group {g+1} - Magnitude - {varname}_{solver_type.upper()}')
    plt.legend()
    plt.ylabel('Normalized amplitude')
    plt.title(title)
    plt.xlabel('Distance from core centre [cm]')
    plt.grid()
    filename = f'{case_name}_{solver_type.upper()}_{varname}_G{g+1}'
    plt.savefig(filename)
    plt.close()
    return filename

def plot_1D_fixed(solver_type, data, x, g, varname=None, case_name=None, title=None, process_data=None):
    plt.clf()
    plt.figure()
    if process_data=='magnitude':
        plt.plot(x, np.abs(data), 'g-', label=f'Group {g+1} - Magnitude - {varname}_{solver_type.upper()}')
        plt.ylabel('Normalized amplitude')
        filename = f'{case_name}_NOISE_magnitude_{varname}_G{g+1}'
    elif process_data=='phase':
        plt.plot(x, np.angle(data), 'g-', label=f'Group {g+1} - Phase - {varname}_{solver_type.upper()}')
        plt.ylabel('Normalized phase')
        filename = f'{case_name}_NOISE_phase_{varname}_G{g+1}'
    plt.title(title)
    plt.xlabel('Distance from core centre [cm]')
    plt.grid()
    plt.savefig(filename)
    plt.close()
    
    return filename

#######################################################################################################
inputs_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'INPUTS'))
sys.path.append(inputs_dir)
from OBJECTIVES1_TEST01_1DMG_CSTest03 import *
sys.path.remove(inputs_dir)

x = globals().get("x")
dx = globals().get("dx")
N = globals().get("N")
group = globals().get("group")
BC = globals().get("BC")
case_name = globals().get("case_name")
solver_type = 'FORWARD'
FLX1_CORESIM = globals().get("FLX1_CORESIM")
FLX2_CORESIM = globals().get("FLX2_CORESIM")
FLX1_ADJ_CORESIM = globals().get("FLX1_ADJ_CORESIM")
FLX2_ADJ_CORESIM = globals().get("FLX2_ADJ_CORESIM")
dFLX1_CORESIM = globals().get("dFLX1_CORESIM")
dFLX2_CORESIM = globals().get("dFLX2_CORESIM")

# Load data from JSON file
with open(f'{case_name}_NOISE_output.json', 'r') as json_file:
    noise_output = json.load(json_file)

# Access keff and PHI from the loaded data
dPHI1 = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI1"]]
dPHI2 = [complex(entry["real"], entry["imaginary"]) for entry in noise_output["dPHI2"]]
dPHI = dPHI1 + dPHI2
dPHI_array = np.array(dPHI)
dPHI_reshaped = dPHI_array.reshape(group, N)

# FLX CORESIM
dFLX1_CORESIM_array = np.array(dFLX1_CORESIM)
dFLX2_CORESIM_array = np.array(dFLX2_CORESIM)
dFLX1_CORESIM_flattened_array = dFLX1_CORESIM_array.ravel()
dFLX2_CORESIM_flattened_array = dFLX2_CORESIM_array.ravel()
dFLX_CORESIM = [[dFLX1_CORESIM_flattened_array], [dFLX2_CORESIM_flattened_array]]
max_dFLX_CORESIM = np.max(np.abs(dFLX_CORESIM))
print(f'Maximum dFLX_CORESIM: {max_dFLX_CORESIM}')
dFLX1_CORESIM_flattened_array = dFLX1_CORESIM_flattened_array / max_dFLX_CORESIM
dFLX2_CORESIM_flattened_array = dFLX2_CORESIM_flattened_array / max_dFLX_CORESIM
dFLX_CORESIM_flattened_array = dFLX_CORESIM / max_dFLX_CORESIM
dFLX_CORESIM_array = np.array(dFLX_CORESIM_flattened_array)
dFLX_CORESIM_reshaped = dFLX_CORESIM_array.reshape(group, N)

# Calculate error and compare
diff_dflx1_CS = (dFLX1_CORESIM_flattened_array - np.array(dPHI1))/dFLX1_CORESIM_flattened_array
diff_dflx2_CS = (dFLX2_CORESIM_flattened_array - np.array(dPHI2))/dFLX2_CORESIM_flattened_array
diff_dflx_CS = [[diff_dflx1_CS], [diff_dflx2_CS]]
diff_dflx_CS_array = np.array(diff_dflx_CS)
diff_dflx_CS_reshaped = diff_dflx_CS_array.reshape(group, N)

#*************************************************************************************
#*************************************************************************************
for g in range(group):
    plot_1D_fixed(solver_type, dFLX_CORESIM_reshaped[g], x, g, varname=f'dFLX_CORESIM', case_name=case_name, title=f'1D Plot of dFLX{g+1}_CORESIM', process_data='magnitude')
    plot_1D_fixed(solver_type, dFLX_CORESIM_reshaped[g], x, g, varname=f'dFLX_CORESIM', case_name=case_name, title=f'1D Plot of dFLX{g+1}_CORESIM', process_data='phase')
    plot_1D_fixed(solver_type, diff_dflx_CS_reshaped[g], x, g, varname=f'diff_dFLX_CORESIM', case_name=case_name, title=f'1D Plot of diff_dFLX{g+1}', process_data='magnitude')
    plot_1D_fixed(solver_type, diff_dflx_CS_reshaped[g], x, g, varname=f'diff_dFLX_CORESIM', case_name=case_name, title=f'1D Plot of diff_dFLX{g+1}', process_data='phase')
