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

def plot_1D_fixed(solver_type, data, x, g, output_dir=None, varname=None, case_name=None, title=None):
    plt.clf()
    plt.figure()
    plt.plot(x, np.abs(data)/np.max(np.abs(data)), 'g-', label=f'Group {g+1} - {varname}_{solver_type.upper()}')
    plt.legend()
    plt.ylabel('Normalized amplitude')
    plt.title(title)
    plt.xlabel('Distance from core centre [cm]')
    plt.grid()
    filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_magnitude_{varname}_G{g+1}'
    plt.savefig(filename)
    plt.close()
    plt.figure()
    plt.plot(x, np.degrees(np.angle(data)), 'g-', label=f'Group {g+1} - Phase - {varname}_{solver_type.upper()}')
    plt.legend()
    plt.ylabel('Normalized amplitude')
    plt.title(f'Magnitude G{g+1}')
    plt.xlabel('Distance from core centre [cm]')
    plt.grid()
    filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_phase_{varname}_G{g+1}'
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
with open(f'{case_name}_ADJOINT_output.json', 'r') as json_file:
    forward_output = json.load(json_file)

# Access keff and PHI from the loaded data
keff = forward_output["keff"]
PHI1_ADJ = forward_output["PHI1_ADJOINT"]
PHI2_ADJ = forward_output["PHI2_ADJOINT"]
PHI_ADJ = PHI1_ADJ + PHI2_ADJ
PHI_ADJ_array = np.array(PHI_ADJ)
PHI_ADJ_reshaped = PHI_ADJ_array.reshape(group, N)

# FLX CORESIM
FLX1_ADJ_CORESIM_array = np.array(FLX1_ADJ_CORESIM)
FLX2_ADJ_CORESIM_array = np.array(FLX2_ADJ_CORESIM)
FLX1_ADJ_CORESIM_flattened_array = FLX1_ADJ_CORESIM_array.ravel()
FLX2_ADJ_CORESIM_flattened_array = FLX2_ADJ_CORESIM_array.ravel()
FLX_ADJ_CORESIM = [[FLX1_ADJ_CORESIM_flattened_array], [FLX2_ADJ_CORESIM_flattened_array]]
FLX_ADJ_CORESIM_array = np.array(FLX_ADJ_CORESIM)
FLX_ADJ_CORESIM_reshaped = FLX_ADJ_CORESIM_array.reshape(group, N)

# Calculate error and compare
diff_flx1_ADJ_CS = np.abs((FLX1_ADJ_CORESIM_flattened_array - np.array(PHI1_ADJ))/FLX1_ADJ_CORESIM_flattened_array) * 100
diff_flx2_ADJ_CS = np.abs((FLX2_ADJ_CORESIM_flattened_array - np.array(PHI2_ADJ))/FLX2_ADJ_CORESIM_flattened_array) * 100
diff_flx_ADJ_CS = [[diff_flx1_ADJ_CS], [diff_flx2_ADJ_CS]]
diff_flx_ADJ_CS_array = np.array(diff_flx_ADJ_CS)
diff_flx_ADJ_CS_reshaped = diff_flx_ADJ_CS_array.reshape(group, N)

#*************************************************************************************
#*************************************************************************************
for g in range(group):
    plot_1D_power(solver_type, FLX_ADJ_CORESIM_reshaped[g], x, g, varname=f'PHI_ADJ', case_name=case_name, title=f'1D Plot of FLX{g+1}_ADJ_CORESIM')
    plot_1D_power(solver_type, diff_flx_ADJ_CS_reshaped[g], x, g, varname=f'diff_flx_ADJ_CS', case_name=case_name, title=f'1D Plot of Relative Difference group {g+1} in %\n Simulator vs CORE SIM+')
