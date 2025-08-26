import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
import h5py

from XSPROCESS_1D_RECT import *
from XSPROCESS_2D_RECT import *
from XSPROCESS_2D_HEXX import *
from XSPROCESS_3D_RECT import *
from XSPROCESS_3D_HEXX import *

##############################################################################
class MatrixBuilderForward1D:
    def __init__(self, group, N, TOT, SIGS_reshaped, BC, dx, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_forward_matrices(self):
        D_mat = FORWARD_D_1D_matrix(self.group, self.BC, self.N, self.dx, self.D)
        TOT_mat = FORWARD_TOT_1D_matrix(self.group, self.N, self.TOT)
        SCAT_mat = FORWARD_SCAT_1D_matrix(self.group, self.N, self.SIGS_reshaped)
        F = FORWARD_NUFIS_1D_matrix(self.group, self.N, self.chi, self.NUFIS)
        M = D_mat + TOT_mat - SCAT_mat
        return M, F

class MatrixBuilderAdjoint1D:
    def __init__(self, group, N, TOT, SIGS_reshaped, BC, dx, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_adjoint_matrices(self):
        D_mat = ADJOINT_D_1D_matrix(self.group, self.BC, self.N, self.dx, self.D)
        TOT_mat = ADJOINT_TOT_1D_matrix(self.group, self.N, self.TOT)
        SCAT_mat = ADJOINT_SCAT_1D_matrix(self.group, self.N, self.SIGS_reshaped)
        F = ADJOINT_NUFIS_1D_matrix(self.group, self.N, self.chi, self.NUFIS)
        M = D_mat + TOT_mat - SCAT_mat
        return M, F

class MatrixBuilderNoise1D:
    def __init__(self, group, N, TOT, SIGS_reshaped, BC, dx, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS):
        self.group = group
        self.N = N
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS
        self.keff = keff
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT = dTOT
        self.dSIGS_reshaped = dSIGS_reshaped
        self.dNUFIS = dNUFIS

    def build_noise_matrices(self):
        chi_p = self.chi
        chi_d = self.chi
        k_complex = 1/self.keff* ((self.l * self.Beff) / (self.l + 1j * self.omega))
        D_mat = NOISE_D_1D_matrix(self.group, self.BC, self.N, self.dx, self.D)
        TOT_mat = NOISE_TOT_1D_matrix(self.group, self.N, self.TOT)
        SCAT_mat = NOISE_SCAT_1D_matrix(self.group, self.N, self.SIGS_reshaped)
        NUFIS_mat = NOISE_NUFIS_1D_matrix(self.group, self.N, chi_p, chi_d, self.NUFIS, k_complex, self.Beff, self.keff)
        FREQ_mat = NOISE_FREQ_1D_matrix(self.group, self.N, self.omega, self.v)
        M = FREQ_mat - D_mat + TOT_mat - NUFIS_mat - SCAT_mat
        dTOT_mat = NOISE_dTOT_1D_matrix(self.group, self.N, self.dTOT)
        dSCAT_mat = NOISE_dSCAT_1D_matrix(self.group, self.N, self.dSIGS_reshaped)
        dNUFIS_mat = NOISE_dNUFIS_1D_matrix(self.group, self.N, chi_p, chi_d, self.dNUFIS, k_complex, self.Beff, self.keff)
        dS = -dTOT_mat + dSCAT_mat + dNUFIS_mat
        return M, dS

##############################################################################
class MatrixBuilderForward2DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_forward_matrices(self):
        D_mat = FORWARD_D_2D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.D)
        TOT_mat = FORWARD_TOT_2D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        SCAT_mat = FORWARD_SCAT_2D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        M = D_mat + TOT_mat - SCAT_mat
        F = FORWARD_NUFIS_2D_rect_matrix(self.group, self.N, self.conv, self.chi, self.NUFIS)
        return M.tocsr(), F.tocsr()

class MatrixBuilderAdjoint2DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_adjoint_matrices(self):
        TOT_mat = ADJOINT_TOT_2D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        SCAT_mat = ADJOINT_SCAT_2D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        D_mat = ADJOINT_D_2D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.D)
        F = ADJOINT_NUFIS_2D_rect_matrix(self.group, self.N, self.conv, self.chi, self.NUFIS)
        M = D_mat + TOT_mat - SCAT_mat
        return M.tocsr(), F.tocsr()

class MatrixBuilderNoise2DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS
        self.keff = keff
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT = dTOT
        self.dSIGS_reshaped = dSIGS_reshaped
        self.dNUFIS = dNUFIS

    def build_noise_matrices(self):
        chi_p = self.chi
        chi_d = self.chi
        k_complex = 1/self.keff* ((self.l * self.Beff) / (self.l + 1j * self.omega))
        D_mat = NOISE_D_2D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.D)
        TOT_mat = NOISE_TOT_2D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        FREQ_mat = NOISE_FREQ_2D_rect_matrix(self.group, self.N, self.conv, self.omega, self.v)
        SCAT_mat = NOISE_SCAT_2D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        NUFIS_mat = NOISE_NUFIS_2D_rect_matrix(self.group, self.N, self.conv, chi_p, chi_d, self.NUFIS, k_complex, self.Beff, self.keff)
        M = FREQ_mat - D_mat + TOT_mat - NUFIS_mat - SCAT_mat
        dTOT_mat = NOISE_dTOT_2D_rect_matrix(self.group, self.N, self.conv, self.dTOT)
        dSCAT_mat = NOISE_dSCAT_2D_rect_matrix(self.group, self.N, self.conv, self.dSIGS_reshaped)
        dNUFIS_mat = NOISE_dNUFIS_2D_rect_matrix(self.group, self.N, self.conv, chi_p, chi_d, self.dNUFIS, k_complex, self.Beff, self.keff)
        dS = -dTOT_mat + dSCAT_mat + dNUFIS_mat
        return M.tocsr(), dS

##############################################################################
class MatrixBuilderForward2DHexx:
    def __init__(self, group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.conv_tri = conv_tri
        self.conv_neighbor = conv_neighbor
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_forward_matrices(self):
        D_hexx_mat = FORWARD_D_2D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor, self.h, self.D, self.level)
        TOT_mat = FORWARD_TOT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.TOT, self.level)
        SCAT_mat = FORWARD_SCAT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.SIGS_reshaped, self.level)
        M = D_hexx_mat + TOT_mat - SCAT_mat
        F = FORWARD_NUFIS_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.chi, self.NUFIS, self.level)
        return M.tocsr(), F.tocsr()

class MatrixBuilderAdjoint2DHexx:
    def __init__(self, group, I_max, J_max, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.conv_tri = conv_tri
        self.conv_neighbor = conv_neighbor
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_adjoint_matrices(self):
        D_hexx_mat = ADJOINT_D_2D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor, self.h, self.D, self.level)
        TOT_mat = ADJOINT_TOT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.TOT, self.level)
        SCAT_mat = ADJOINT_SCAT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.SIGS_reshaped, self.level)
        M = D_hexx_mat + TOT_mat - SCAT_mat
        F = ADJOINT_NUFIS_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.chi, self.NUFIS, self.level)
        return M.tocsr(), F.tocsr()

class MatrixBuilderNoise2DHexx:
    def __init__(self, group, I_max, J_max, N_hexx, conv_tri, conv_neighbor, TOT, SIGS_reshaped, BC, h, level, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, chi_hexx, dNUFIS_hexx, noise_section, type_noise):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.conv_tri = conv_tri
        self.conv_neighbor = conv_neighbor
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS
        self.keff = keff
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT_hexx = dTOT_hexx
        self.dSIGS_hexx = dSIGS_hexx
        self.chi_hexx = chi_hexx
        self.dNUFIS_hexx = dNUFIS_hexx
        self.noise_section = noise_section
        self.type_noise = type_noise
        self.N_hexx = N_hexx

    def build_noise_matrices(self):
        chi_p = self.chi
        chi_d = self.chi
        chi_p_hexx = self.chi_hexx
        chi_d_hexx = self.chi_hexx
        k_complex = 1/self.keff* ((self.l * self.Beff) / (self.l + 1j * self.omega))
        D_hexx_mat = NOISE_D_2D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor, self.h, self.D, self.level)
        TOT_mat = NOISE_TOT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.TOT, self.level)
        FREQ_mat = NOISE_FREQ_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.omega, self.v, self.level)
        SCAT_mat = NOISE_SCAT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.SIGS_reshaped, self.level)
        NUFIS_mat = NOISE_NUFIS_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, chi_p, chi_d, self.NUFIS, k_complex, self.Beff, self.keff, self.level)
        M = FREQ_mat - D_hexx_mat + TOT_mat - NUFIS_mat - SCAT_mat

        dTOT_mat = NOISE_dTOT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.dTOT_hexx, self.level)
        dSCAT_mat = NOISE_dSCAT_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, self.dSIGS_hexx, self.level)
        dNUFIS_mat = NOISE_dNUFIS_2D_hexx_matrix(self.group, self.I_max, self.J_max, self.conv_tri, chi_p_hexx, chi_d_hexx, self.dNUFIS_hexx, k_complex, self.Beff, self.keff, self.level)
        dS = -dTOT_mat + dSCAT_mat + dNUFIS_mat
        return M.tocsr(), dS

##############################################################################
class MatrixBuilderForward3DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_forward_matrices(self):
        D_mat = FORWARD_D_3D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.dz, self.D)
        TOT_mat = FORWARD_TOT_3D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        SCAT_mat = FORWARD_SCAT_3D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        M = D_mat + TOT_mat - SCAT_mat
        F = FORWARD_NUFIS_3D_rect_matrix(self.group, self.N, self.conv, self.chi, self.NUFIS)
        return M.tocsr(), F.tocsr()

class MatrixBuilderAdjoint3DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_adjoint_matrices(self):
        TOT_mat = ADJOINT_TOT_3D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        SCAT_mat = ADJOINT_SCAT_3D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        D_mat = ADJOINT_D_3D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.dz, self.D)
        F = ADJOINT_NUFIS_3D_rect_matrix(self.group, self.N, self.conv, self.chi, self.NUFIS)
        M = D_mat + TOT_mat - SCAT_mat
        return M.tocsr(), F.tocsr()

class MatrixBuilderNoise3DRect:
    def __init__(self, group, N, conv, TOT, SIGS_reshaped, BC, dx, dy, dz, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT, dSIGS_reshaped, dNUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS
        self.keff = keff
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT = dTOT
        self.dSIGS_reshaped = dSIGS_reshaped
        self.dNUFIS = dNUFIS

    def build_noise_matrices(self):
        chi_p = self.chi
        chi_d = self.chi
        k_complex = 1/self.keff* ((self.l * self.Beff) / (self.l + 1j * self.omega))
        D_mat = NOISE_D_3D_rect_matrix(self.group, self.BC, self.conv, self.dx, self.dy, self.dz, self.D)
        TOT_mat = NOISE_TOT_3D_rect_matrix(self.group, self.N, self.conv, self.TOT)
        FREQ_mat = NOISE_FREQ_3D_rect_matrix(self.group, self.N, self.conv, self.omega, self.v)
        SCAT_mat = NOISE_SCAT_3D_rect_matrix(self.group, self.N, self.conv, self.SIGS_reshaped)
        NUFIS_mat = NOISE_NUFIS_3D_rect_matrix(self.group, self.N, self.conv, chi_p, chi_d, self.NUFIS, k_complex, self.Beff, self.keff)
        M = FREQ_mat - D_mat + TOT_mat - NUFIS_mat - SCAT_mat
        dTOT_mat = NOISE_dTOT_3D_rect_matrix(self.group, self.N, self.conv, self.dTOT)
        dSCAT_mat = NOISE_dSCAT_3D_rect_matrix(self.group, self.N, self.conv, self.dSIGS_reshaped)
        dNUFIS_mat = NOISE_dNUFIS_3D_rect_matrix(self.group, self.N, self.conv, chi_p, chi_d, self.dNUFIS, k_complex, self.Beff, self.keff)
        dS = -dTOT_mat + dSCAT_mat + dNUFIS_mat
        return M.tocsr(), dS

##############################################################################
class MatrixBuilderForward3DHexx:
    def __init__(self, group, I_max, J_max, K_max, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.K_max = K_max
        self.conv_tri = conv_tri
        self.conv_neighbor_3D = conv_neighbor_3D
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.dz = dz
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_forward_matrices(self):
        D_hexx_mat = FORWARD_D_3D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor_3D, self.h, self.dz, self.D, self.level)
        TOT_mat = FORWARD_TOT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.TOT, self.level)
        SCAT_mat = FORWARD_SCAT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.SIGS_reshaped, self.level)
        M = D_hexx_mat + TOT_mat - SCAT_mat
        F = FORWARD_NUFIS_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.chi, self.NUFIS, self.level)
        return M.tocsr(), F.tocsr()

class MatrixBuilderAdjoint3DHexx:
    def __init__(self, group, I_max, J_max, K_max, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.K_max = K_max
        self.conv_tri = conv_tri
        self.conv_neighbor_3D = conv_neighbor_3D
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.dz = dz
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS

    def build_adjoint_matrices(self):
        D_hexx_mat = ADJOINT_D_3D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor_3D, self.h, self.dz, self.D, self.level)
        TOT_mat = ADJOINT_TOT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.TOT, self.level)
        SCAT_mat = ADJOINT_SCAT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.SIGS_reshaped, self.level)
        M = D_hexx_mat + TOT_mat - SCAT_mat
        F = ADJOINT_NUFIS_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.chi, self.NUFIS, self.level)
        return M.tocsr(), F.tocsr()

class MatrixBuilderNoise3DHexx:
    def __init__(self, group, I_max, J_max, K_max, N_hexx, conv_tri, conv_neighbor_3D, TOT, SIGS_reshaped, BC, h, dz, level, D, chi, NUFIS, keff, v, Beff, omega, l, dTOT_hexx, dSIGS_hexx, chi_hexx, dNUFIS_hexx, noise_section, type_noise):
        self.group = group
        self.I_max = I_max
        self.J_max = J_max
        self.K_max = K_max
        self.conv_tri = conv_tri
        self.conv_neighbor_3D = conv_neighbor_3D
        self.TOT = TOT
        self.SIGS_reshaped = SIGS_reshaped
        self.BC = BC
        self.h = h
        self.dz = dz
        self.level = level
        self.D = D
        self.chi = chi
        self.NUFIS = NUFIS
        self.keff = keff
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT_hexx = dTOT_hexx
        self.dSIGS_hexx = dSIGS_hexx
        self.chi_hexx = chi_hexx
        self.dNUFIS_hexx = dNUFIS_hexx
        self.noise_section = noise_section
        self.type_noise = type_noise
        self.N_hexx = N_hexx

    def build_noise_matrices(self):
        chi_p = self.chi
        chi_d = self.chi
        chi_p_hexx = self.chi_hexx
        chi_d_hexx = self.chi_hexx
        k_complex = 1/self.keff* ((self.l * self.Beff) / (self.l + 1j * self.omega))
        print(k_complex)
        D_hexx_mat = NOISE_D_3D_hexx_matrix(self.group, self.BC, self.conv_tri, self.conv_neighbor_3D, self.h, self.dz, self.D, self.level)
        TOT_mat = NOISE_TOT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.TOT, self.level)
        FREQ_mat = NOISE_FREQ_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.omega, self.v, self.level)
        SCAT_mat = NOISE_SCAT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.SIGS_reshaped, self.level)
        NUFIS_mat = NOISE_NUFIS_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, chi_p, chi_d, self.NUFIS, k_complex, self.Beff, self.keff, self.level)
        M = FREQ_mat - D_hexx_mat + TOT_mat - NUFIS_mat - SCAT_mat

        dTOT_mat = NOISE_dTOT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.dTOT_hexx, self.level)
        dSCAT_mat = NOISE_dSCAT_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, self.dSIGS_hexx, self.level)
        dNUFIS_mat = NOISE_dNUFIS_3D_hexx_matrix(self.group, self.K_max, self.J_max, self.I_max, self.conv_tri, chi_p_hexx, chi_d_hexx, self.dNUFIS_hexx, k_complex, self.Beff, self.keff, self.level)
        dS = -dTOT_mat + dSCAT_mat + dNUFIS_mat
        return M.tocsr(), dS

######################################################
class MatrixFreeM2DRect:
    """
    Matrix-free multigroup diffusion operator on a 2D rect grid.

    - Supports BC codes: 1=Dirichlet (vacuum), 2=Reflective, 3=Robin/Marshak (vacuum-like).
    - Diagonal is implemented for Jacobi PC via getDiagonal().
    - Uses conv[] to map active cells to compact unknown ordering (1-based indices).
    """

    def __init__(self, group, N, conv, dx, dy, D, TOT, SIGS, BC, I_max, J_max):
        self.group = group
        self.N = N
        self.conv = conv
        self.dx, self.dy = dx, dy
        self.D = D          # shape [group][J][I]
        self.TOT = TOT      # shape [group][N_flat]
        self.SIGS = SIGS    # shape [group][group][N_flat]
        self.BC = BC        # (north, south, east, west, top, bottom) — top/bottom unused in 2D
        self.I_max, self.J_max = I_max, J_max

        # Cache
        self.max_conv = max(self.conv)

    # --------- coefficient helpers (shared by mult() and getDiagonal()) ---------
    def _DIFXCOEF(self, D_west, D_mid, D_east, dx):
        a1 = (2*D_mid*D_west)/((D_west + D_mid) * dx*dx)
        a2 = a1 + (2*D_east*D_mid)/((D_east + D_mid) * dx*dx)
        a3 = (2*D_east*D_mid)/((D_east + D_mid) * dx*dx)
        return a1, a2, a3

    def _DIFXCOEF_WB(self, D_mid, D_east, dx, BC_west):
        if BC_west == 1:   # Dirichlet/vacuum
            a2 = (2*D_mid)/(dx*dx) + (2*D_east*D_mid)/((D_east + D_mid) * dx*dx)
        elif BC_west == 2: # Reflective
            a2 = (2*D_east*D_mid)/((D_east + D_mid) * dx*dx)
        elif BC_west == 3: # Robin/Marshak
            a2 = (2*D_mid)/((4*D_mid*dx) + dx*dx) + (2*D_east*D_mid)/((D_east + D_mid) * dx*dx)
        a3 = (2*D_east*D_mid)/((D_east + D_mid) * dx*dx)
        return a2, a3

    def _DIFXCOEF_EB(self, D_west, D_mid, dx, BC_east):
        if BC_east == 1:
            a2 = (2*D_mid)/(dx*dx) + (2*D_west*D_mid)/((D_west + D_mid) * dx*dx)
        elif BC_east == 2:
            a2 = (2*D_west*D_mid)/((D_west + D_mid) * dx*dx)
        elif BC_east == 3:
            a2 = (2*D_mid)/((4*D_mid*dx) + dx*dx) + (2*D_mid*D_west)/((D_west + D_mid) * dx*dx)
        a1 = (2*D_mid*D_west)/((D_west + D_mid) * dx*dx)
        return a1, a2

    def _DIFYCOEF(self, D_bot, D_mid, D_top, dy):
        b1 = (2*D_mid*D_bot)/((D_bot + D_mid) * dy*dy)
        b2 = b1 + (2*D_top*D_mid)/((D_top + D_mid) * dy*dy)
        b3 = (2*D_top*D_mid)/((D_top + D_mid) * dy*dy)
        return b1, b2, b3

    def _DIFYCOEF_SB(self, D_mid, D_top, dy, BC_south):
        if BC_south == 1:
            b2 = (2*D_mid)/(dy*dy) + (2*D_top*D_mid)/((D_top + D_mid) * dy*dy)
        elif BC_south == 2:
            b2 = (2*D_top*D_mid)/((D_top + D_mid) * dy*dy)
        elif BC_south == 3:
            b2 = (2*D_mid)/((4*D_mid*dy) + dy*dy) + (2*D_top*D_mid)/((D_top + D_mid) * dy*dy)
        b3 = (2*D_top*D_mid)/((D_top + D_mid) * dy*dy)
        return b2, b3

    def _DIFYCOEF_NB(self, D_bot, D_mid, dy, BC_north):
        if BC_north == 1:
            b2 = (2*D_mid)/(dy*dy) + (2*D_mid*D_bot)/((D_bot + D_mid) * dy*dy)
        elif BC_north == 2:
            b2 = (2*D_mid*D_bot)/((D_bot + D_mid) * dy*dy)
        elif BC_north == 3:
            b2 = (2*D_mid)/((4*D_mid*dy) + dy*dy) + (2*D_mid*D_bot)/((D_bot + D_mid) * dy*dy)
        b1 = (2*D_mid*D_bot)/((D_bot + D_mid) * dy*dy)
        return b1, b2

    # --------- matrix-free mat-vec ---------
    def mult(self, A, x, y):
        # PETSc 3.23+ style
        x_arr = x.getArray(readonly=True)   # read-only view
        y_arr = y.getArray()                # writable view
        y_arr[:] = 0.0

        max_conv = self.max_conv
        I_max, J_max = self.I_max, self.J_max
        dx, dy = self.dx, self.dy
        BC_north, BC_south, BC_east, BC_west = self.BC

        for g in range(self.group):
            phi_g = x_arr[g*max_conv:(g+1)*max_conv]

            for j in range(J_max):
                for i in range(I_max):
                    m = j * I_max + i
                    if self.D[g][j][i] == 0:
                        continue

                    row = g*max_conv + (self.conv[m]-1)

                    # ----- X-direction -----
                    if i == 0 or (i > 0 and self.D[g][j][i-1] == 0):
                        # west boundary/block
                        a2, a3 = self._DIFXCOEF_WB(self.D[g][j][i],
                                                    self.D[g][j][i+1], dx, BC_west)
                        y_arr[row] += a2 * phi_g[self.conv[m]-1]
                        if i < I_max-1 and self.D[g][j][i+1] != 0:
                            y_arr[row] -= a3 * phi_g[self.conv[m+1]-1]
                    elif i == I_max-1 or (i < I_max-1 and self.D[g][j][i+1] == 0):
                        # east boundary/block
                        a1, a2 = self._DIFXCOEF_EB(self.D[g][j][i-1],
                                                    self.D[g][j][i], dx, BC_east)
                        y_arr[row] += a2 * phi_g[self.conv[m]-1]
                        if i > 0 and self.D[g][j][i-1] != 0:
                            y_arr[row] -= a1 * phi_g[self.conv[m-1]-1]
                    else:
                        a1, a2, a3 = self._DIFXCOEF(self.D[g][j][i-1],
                                                     self.D[g][j][i],
                                                     self.D[g][j][i+1], dx)
                        if i > 0 and self.D[g][j][i-1] != 0:
                            y_arr[row] -= a1 * phi_g[self.conv[m-1]-1]
                        y_arr[row] += a2 * phi_g[self.conv[m]-1]
                        if i < I_max-1 and self.D[g][j][i+1] != 0:
                            y_arr[row] -= a3 * phi_g[self.conv[m+1]-1]

                    # ----- Y-direction -----
                    # NOTE: vertical neighbors are at m ± I_max
                    if j == 0 or (j > 0 and self.D[g][j-1][i] == 0):
                        # south boundary/block
                        b2, b3 = self._DIFYCOEF_SB(self.D[g][j][i],
                                                    self.D[g][j+1][i], dy, BC_south)
                        y_arr[row] += b2 * phi_g[self.conv[m]-1]
                        if j < J_max-1 and self.D[g][j+1][i] != 0:
                            y_arr[row] -= b3 * phi_g[self.conv[m + I_max]-1]
                    elif j == J_max-1 or (j < J_max-1 and self.D[g][j+1][i] == 0):
                        # north boundary/block
                        b1, b2 = self._DIFYCOEF_NB(self.D[g][j-1][i],
                                                    self.D[g][j][i], dy, BC_north)
                        y_arr[row] += b2 * phi_g[self.conv[m]-1]
                        if j > 0 and self.D[g][j-1][i] != 0:
                            y_arr[row] -= b1 * phi_g[self.conv[m - I_max]-1]
                    else:
                        b1, b2, b3 = self._DIFYCOEF(self.D[g][j-1][i],
                                                    self.D[g][j][i],
                                                    self.D[g][j+1][i], dy)
                        if j > 0 and self.D[g][j-1][i] != 0:
                            y_arr[row] -= b1 * phi_g[self.conv[m - I_max]-1]
                        y_arr[row] += b2 * phi_g[self.conv[m]-1]
                        if j < J_max-1 and self.D[g][j+1][i] != 0:
                            y_arr[row] -= b3 * phi_g[self.conv[m + I_max]-1]

                    # ----- total & scattering -----
                    y_arr[row] += self.TOT[g][m] * phi_g[self.conv[m]-1]

                    # off-diagonal group scattering: minus sign (coupling to other groups)
                    for h in range(self.group):
                        if h == g:
                            # If you include self-scatter in TOT, keep this as continue.
                            # If not, you can subtract SIGS[g][g][m] here.
                            continue
                        phi_h = x_arr[h*max_conv:(h+1)*max_conv]
                        y_arr[row] -= self.SIGS[g][h][m] * phi_h[self.conv[m]-1]

        # no restoreArray()/setArray() needed — we wrote directly into y_arr

    # --------- diagonal for Jacobi PC ---------
    def getDiagonal(self, A, diag):
        """
        Fill PETSc Vec 'diag' with diagonal entries of the operator.
        """
        diag_arr = diag.getArray()
        diag_arr[:] = 0.0

        max_conv = self.max_conv
        I_max, J_max = self.I_max, self.J_max
        dx, dy = self.dx, self.dy
        BC_north, BC_south, BC_east, BC_west = self.BC

        for g in range(self.group):
            for j in range(J_max):
                for i in range(I_max):
                    m = j * I_max + i
                    if self.D[g][j][i] == 0:
                        continue

                    row = g*max_conv + (self.conv[m]-1)
                    Dmid = self.D[g][j][i]

                    # X self-term
                    if i == 0 or (i > 0 and self.D[g][j][i-1] == 0):
                        a2, _ = self._DIFXCOEF_WB(Dmid, self.D[g][j][i+1], dx, BC_west)
                        diag_arr[row] += a2
                    elif i == I_max-1 or (i < I_max-1 and self.D[g][j][i+1] == 0):
                        _, a2 = self._DIFXCOEF_EB(self.D[g][j][i-1], Dmid, dx, BC_east)
                        diag_arr[row] += a2
                    else:
                        _, a2, _ = self._DIFXCOEF(self.D[g][j][i-1], Dmid, self.D[g][j][i+1], dx)
                        diag_arr[row] += a2

                    # Y self-term
                    if j == 0 or (j > 0 and self.D[g][j-1][i] == 0):
                        b2, _ = self._DIFYCOEF_SB(Dmid, self.D[g][j+1][i], dy, BC_south)
                        diag_arr[row] += b2
                    elif j == J_max-1 or (j < J_max-1 and self.D[g][j+1][i] == 0):
                        _, b2 = self._DIFYCOEF_NB(self.D[g][j-1][i], Dmid, dy, BC_north)
                        diag_arr[row] += b2
                    else:
                        _, b2, _ = self._DIFYCOEF(self.D[g][j-1][i], Dmid, self.D[g][j+1][i], dy)
                        diag_arr[row] += b2

                    # total removal
                    diag_arr[row] += self.TOT[g][m]

                    # If your TOT already includes self-scatter, do nothing.
                    # If not, subtract self-scatter here:
                    # diag_arr[row] -= self.SIGS[g][g][m]
        # done; diag_arr is a live view into PETSc Vec

class MatrixFreeF2DRect:
    def __init__(self, group, N, conv, chi, NUFIS):
        self.group = group
        self.N = N
        self.conv = conv
        self.chi = chi
        self.NUFIS = NUFIS

    def mult(self, A, x, y):
        # Get views
        x_arr = x.getArray(readonly=True)  # read-only
        y_arr = y.getArray()               # writable
        y_arr[:] = 0.0                      # clear

        max_conv = max(self.conv)

        # Loop groups: produce νΣf φ and distribute with χ
        for j in range(self.group):
            phi_j = x_arr[j*max_conv:(j+1)*max_conv]
            for g in range(self.group):
                for n in range(self.N):
                    row = g*max_conv + (self.conv[n]-1)
                    y_arr[row] += self.chi[g][n] * self.NUFIS[j][n] * phi_j[self.conv[n]-1]

        # no restoreArray() needed

