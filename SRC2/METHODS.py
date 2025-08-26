import numpy as np
from scipy.sparse.linalg import spilu, LinearOperator, spsolve, gmres, splu, cg
from scipy.integrate import trapezoid
from petsc4py import PETSc

from MATRIX_BUILDER import MatrixFreeM2DRect, MatrixFreeF2DRect

class PowerMethodSolver1D:
    def __init__(self, group, N, M, F, x, precond, tol=1e-06):
        self.group = group
        self.N = N
        self.M = M
        self.F = F
        self.x = x
        self.precond = precond
        self.tol = tol

    def solve(self):
        phi = np.ones(self.group * self.N)
        keff = 1.0
        errflux = errkeff = self.tol + 1
        iter_count = 0
        x_integrate = np.tile(self.x, self.group)

        if self.precond == 1:
            print(f'Solving using ILU')
            M_csc = self.M.tocsc()
            ilu = spilu(M_csc)
            M_preconditioner = LinearOperator(M_csc.shape, matvec=ilu.solve)
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            M_csc = self.M.tocsc()
            lu = splu(M_csc)
            M_preconditioner = LinearOperator(M_csc.shape, matvec=lu.solve)
        else:
            print(f'Solving using Sparse Solver')

        while errflux > self.tol:
            phi_old = phi.copy()
            k_old = keff

            S = 1 / k_old * (self.F @ phi_old)
            if self.precond == 1:
                phi, info = gmres(M_csc, S, rtol=1e-8, maxiter=1000, M=M_preconditioner)
            elif self.precond == 2:
                phi = lu.solve(S)
            else:
                phi = spsolve(self.M, S)

            # Update keff
            keff = k_old * trapezoid(self.F @ phi, x=x_integrate, axis=0) / \
                   trapezoid(self.F @ phi_old, x=x_integrate, axis=0)

            residual = S - self.M.dot(phi)
            residual_norm = np.linalg.norm(residual)

            # Normalization
            phi /= np.max(phi)

            # Calculate errors
            errkeff = np.abs((keff - k_old) / k_old)
            errflux = np.max(np.abs(phi - phi_old) / (np.abs(phi) + 1E-20))

            iter_count += 1
            print(f'Iteration: {iter_count}, keff = {keff:.5f}, errkeff = {errkeff:.6e}, '
                  f'errflux = {errflux:.5e}, residual = {residual_norm:.5e}')

        return keff, phi

class FixedSourceSolver1D:
    def __init__(self, group, N, M, dS, dSOURCE, PHI, precond, tol=1e-06):
        self.group = group
        self.N = N
        self.M = M
        self.dS = dS
        self.precond = precond
        self.tol = tol
        self.PHI = PHI
        self.dSOURCE = dSOURCE

    def solve(self):
        dPHI = np.ones(self.group * self.N, dtype=complex)
        errdPHI = 1
        tol = 1e-06
        iter = 0
        self.dSOURCE = [item for sublist in self.dSOURCE for item in sublist] if all(isinstance(sublist, list) for sublist in self.dSOURCE) else self.dSOURCE #[item for sublist in self.dSOURCE for item in sublist]

        if self.precond == 1:
            print('Solving using ILU')
            M_csc = self.M.tocsc()
            ilu = spilu(M_csc)
            M_preconditioner = LinearOperator(M_csc.shape, matvec=ilu.solve)
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            M_csc = self.M.tocsc()
            lu = splu(M_csc)
            M_preconditioner = LinearOperator(M_csc.shape, matvec=lu.solve)
        else:
            print('Solving using Solver')

        while errdPHI > self.tol:
            dPHI_old = dPHI.copy()

            # Set up RHS
            S = self.dS.dot(self.PHI) + self.dSOURCE

            if self.precond == 1:
                dPHI, info = cg(M_csc, S, rtol=1e-8, maxiter=1000, M=M_preconditioner)
            elif self.precond == 2:
                dPHI = lu.solve(S)
            else:
                dPHI = spsolve(self.M, S)

            # Calculate errors
            errdPHI = np.max(np.abs(dPHI - dPHI_old) / (np.abs(dPHI) + 1E-20))

            iter += 1
            print(f'Iteration: {iter}, errflux = {errdPHI:.6e}')

        return dPHI

class PowerMethodSolver2DRect:
    def __init__(self, group, N, conv, M, F, dx, dy, precond, tol):
        self.group = group
        self.N = N
        self.M = M
        self.F = F
        self.dx = dx
        self.dy = dy
        self.tol = tol
        self.precond = precond
        self.conv = conv

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data))
        F_petsc = PETSc.Mat().createAIJ(size=self.F.shape, csr=(self.F.indptr, self.F.indices, self.F.data))
        M_petsc.assemble()
        F_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        phi_temp = np.ones(self.group * max(self.conv))
        keff = 1.0
        errflux = errkeff = 1.0
        iter_count = 0

        while errflux > self.tol and errkeff > self.tol:
            phi_temp_old = phi_temp.copy()
            k_old = keff

            S = 1 / k_old * (self.F @ phi_temp_old)
            # PETSc Vectors for RHS and solution
            S_petsc = PETSc.Vec().createWithArray(S)
            phi_temp_petsc = PETSc.Vec().createWithArray(phi_temp)

            # Solve the linear system using PETSc KSP
            ksp.solve(S_petsc, phi_temp_petsc)

            # Get result back into NumPy array
            phi_temp = phi_temp_petsc.getArray()

            # Update keff
            keff = k_old * trapezoid(self.F @ phi_temp, dx=self.dx * self.dy, axis=0) / \
                   trapezoid(self.F @ phi_temp_old, dx=self.dx * self.dy, axis=0)

            residual = S - self.M.dot(phi_temp)
            residual_norm = np.linalg.norm(residual)

            # Normalization
            phi_temp /= np.max(phi_temp)

            # Calculate errors
            errkeff = np.abs((keff - k_old) / k_old)
            errflux = np.max(np.abs(phi_temp - phi_temp_old) / (np.abs(phi_temp) + 1E-20))

            iter_count += 1
            print(f'Iteration: {iter_count}, keff = {keff:.5f}, errkeff = {errkeff:.6e}, '
                  f'errflux = {errflux:.5e}, residual = {residual_norm:.5e}')

        return keff, phi_temp

class FixedSourceSolver2DRect:
    def __init__(self, group, N, conv, M, dS, PHI, dx, dy, precond, tol):
        self.group = group
        self.N = N
        self.M = M
        self.dS = dS
        self.dx = dx
        self.dy = dy
        self.tol = tol
        self.conv = conv
        self.PHI = PHI
        self.precond = precond

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data), comm=PETSc.COMM_WORLD)
        M_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        dPHI_temp = np.ones(self.group*max(self.conv), dtype=complex)
        errdPHI = 1
        iter = 0

        while errdPHI > self.tol:
            dPHI_tempold = dPHI_temp.copy()

            # Set up RHS
            S = self.dS.dot(self.PHI)

            # PETSc Vectors for RHS and solution
            S_petsc = PETSc.Vec().createWithArray(S)
            dPHI_temp_petsc = PETSc.Vec().createWithArray(dPHI_temp)

            # Solve the linear system using PETSc KSP
            ksp.solve(S_petsc, dPHI_temp_petsc)

            # Get result back into NumPy array
            dPHI_temp = dPHI_temp_petsc.getArray()

            # Calculate errors
            errdPHI = np.max(np.abs(dPHI_temp - dPHI_tempold) / (np.abs(dPHI_temp) + 1E-20))

            iter += 1
            print(f'Iteration: {iter}, errflux = {errdPHI:.6e}')

        return dPHI_temp

class PowerMethodSolver2DHexx:
    def __init__(self, group, conv_tri, M, F, h, precond, tol):
        self.group = group
        self.M = M
        self.F = F
        self.h = h
        self.tol = tol
        self.precond = precond
        self.conv_tri = conv_tri

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data))
        F_petsc = PETSc.Mat().createAIJ(size=self.F.shape, csr=(self.F.indptr, self.F.indices, self.F.data))
        M_petsc.assemble()
        F_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        phi_temp = np.ones(self.group * max(self.conv_tri))
        keff = 1.0
        errflux = errkeff = 1.0
        iter_count = 0

        while errflux > self.tol and errkeff > self.tol:
            phi_temp_old = phi_temp.copy()
            k_old = keff

            S = 1 / k_old * (self.F @ phi_temp_old)
            # PETSc Vectors for RHS and solution
            S_petsc = PETSc.Vec().createWithArray(S)
            phi_temp_petsc = PETSc.Vec().createWithArray(phi_temp)

            # Solve the linear system using PETSc KSP
            ksp.solve(S_petsc, phi_temp_petsc)

            # Get result back into NumPy array
            phi_temp = phi_temp_petsc.getArray()

            # Update keff
            keff = k_old * trapezoid(self.F @ phi_temp, dx=self.h**2/4*np.sqrt(3), axis=0) / \
                   trapezoid(self.F @ phi_temp_old, dx=self.h**2/4*np.sqrt(3), axis=0)

            residual = S - self.M.dot(phi_temp)
            residual_norm = np.linalg.norm(residual)

            # Normalization
            phi_temp /= np.max(phi_temp)

            # Calculate errors
            errkeff = np.abs((keff - k_old) / k_old)
            errflux = np.max(np.abs(phi_temp - phi_temp_old) / (np.abs(phi_temp) + 1E-20))

            iter_count += 1
            print(f'Iteration: {iter_count}, keff = {keff:.5f}, errkeff = {errkeff:.6e}, '
                  f'errflux = {errflux:.5e}, residual = {residual_norm:.5e}')

        return keff, phi_temp

class FixedSourceSolver2DHexx:
    def __init__(self, group, conv_tri, M, dS, PHI, precond, tol):
        self.group = group
        self.M = M
        self.dS = dS
        self.tol = tol
        self.conv_tri = conv_tri
        self.PHI = PHI
        self.precond = precond

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data), comm=PETSc.COMM_WORLD)
        M_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        dPHI_temp = np.ones(self.group*max(self.conv_tri), dtype=complex)
        errdPHI = 1
        iter = 0

        while errdPHI > self.tol:
            dPHI_tempold = dPHI_temp.copy()

            # Set up RHS
            S = self.dS.dot(self.PHI)

            # PETSc Vectors for RHS and solution
            S_petsc = PETSc.Vec().createWithArray(S)
            dPHI_temp_petsc = PETSc.Vec().createWithArray(dPHI_temp)

            # Solve the linear system using PETSc KSP
            ksp.solve(S_petsc, dPHI_temp_petsc)

            # Get result back into NumPy array
            dPHI_temp = dPHI_temp_petsc.getArray()

            # Calculate errors
            errdPHI = np.max(np.abs(dPHI_temp - dPHI_tempold) / (np.abs(dPHI_temp) + 1E-20))

            iter += 1
            print(f'Iteration: {iter}, errflux = {errdPHI:.6e}')

        return dPHI_temp

class PowerMethodSolver3DRect:
    def __init__(self, group, N, conv, M, F, dx, dy, dz, precond, tol):
        self.group = group
        self.N = N
        self.M = M
        self.F = F
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.tol = tol
        self.precond = precond
        self.conv = conv

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data))
        F_petsc = PETSc.Mat().createAIJ(size=self.F.shape, csr=(self.F.indptr, self.F.indices, self.F.data))
        M_petsc.assemble()
        F_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        phi_temp = np.ones(self.group * max(self.conv))
        keff = 1.0
        errflux = errkeff = 1.0
        iter_count = 0

        while errflux > self.tol and errkeff > self.tol:
            phi_temp_old = phi_temp.copy()
            k_old = keff

            S = 1 / k_old * (self.F @ phi_temp_old)
            # PETSc Vectors for RHS and solution
            S_petsc = PETSc.Vec().createWithArray(S)
            phi_temp_petsc = PETSc.Vec().createWithArray(phi_temp)

            # Solve the linear system using PETSc KSP
            ksp.solve(S_petsc, phi_temp_petsc)

            # Get result back into NumPy array
            phi_temp = phi_temp_petsc.getArray()

            # Update keff
            keff = k_old * trapezoid(self.F @ phi_temp, dx=self.dx * self.dy * self.dz, axis=0) / \
                   trapezoid(self.F @ phi_temp_old, dx=self.dx * self.dy * self.dz, axis=0)

            residual = S - self.M.dot(phi_temp)
            residual_norm = np.linalg.norm(residual)

            # Normalization
            phi_temp /= np.max(phi_temp)

            # Calculate errors
            errkeff = np.abs((keff - k_old) / k_old)
            errflux = np.max(np.abs(phi_temp - phi_temp_old) / (np.abs(phi_temp) + 1E-20))

            iter_count += 1
            print(f'Iteration: {iter_count}, keff = {keff:.5f}, errkeff = {errkeff:.6e}, '
                  f'errflux = {errflux:.5e}, residual = {residual_norm:.5e}')

        return keff, phi_temp

class FixedSourceSolver3DRect:
    def __init__(self, group, N, conv, M, dS, PHI, dx, dy, dz, precond, tol):
        self.group = group
        self.N = N
        self.M = M
        self.dS = dS
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.tol = tol
        self.conv = conv
        self.PHI = PHI
        self.precond = precond

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data), comm=PETSc.COMM_WORLD)
        M_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        dPHI_temp = np.ones(self.group*max(self.conv), dtype=complex)
        errdPHI = 1
        iter = 0

        while errdPHI > self.tol:
            dPHI_tempold = dPHI_temp.copy()

            # Set up RHS
            S = self.dS.dot(self.PHI)

            # PETSc Vectors for RHS and solution
            S_petsc = PETSc.Vec().createWithArray(S)
            dPHI_temp_petsc = PETSc.Vec().createWithArray(dPHI_temp)

            # Solve the linear system using PETSc KSP
            ksp.solve(S_petsc, dPHI_temp_petsc)

            # Get result back into NumPy array
            dPHI_temp = dPHI_temp_petsc.getArray()

            # Calculate errors
            errdPHI = np.max(np.abs(dPHI_temp - dPHI_tempold) / (np.abs(dPHI_temp) + 1E-20))

            iter += 1
            print(f'Iteration: {iter}, errflux = {errdPHI:.6e}')

        return dPHI_temp

class PowerMethodSolver3DHexx:
    def __init__(self, group, conv_tri, M, F, h, dz, precond, tol):
        self.group = group
        self.M = M
        self.F = F
        self.h = h
        self.dz = dz
        self.tol = tol
        self.precond = precond
        self.conv_tri = conv_tri

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data))
        F_petsc = PETSc.Mat().createAIJ(size=self.F.shape, csr=(self.F.indptr, self.F.indices, self.F.data))
        M_petsc.assemble()
        F_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Iteration for Power Method
        phi_temp = np.ones(self.group * max(self.conv_tri))
        keff = 1.0
        errflux = errkeff = 1.0
        iter_count = 0

        while errflux > self.tol and errkeff > self.tol:
            phi_temp_old = phi_temp.copy()
            k_old = keff

            S = 1 / k_old * (self.F @ phi_temp_old)
            # PETSc Vectors for RHS and solution
            S_petsc = PETSc.Vec().createWithArray(S)
            phi_temp_petsc = PETSc.Vec().createWithArray(phi_temp)

            # Solve the linear system using PETSc KSP
            ksp.solve(S_petsc, phi_temp_petsc)

            # Get result back into NumPy array
            phi_temp = phi_temp_petsc.getArray()

            # Update keff
            keff = k_old * trapezoid(self.F @ phi_temp, dx=self.h**2/4*np.sqrt(3)*self.dz, axis=0) / \
                   trapezoid(self.F @ phi_temp_old, dx=self.h**2/4*np.sqrt(3)*self.dz, axis=0)

            residual = S - self.M.dot(phi_temp)
            residual_norm = np.linalg.norm(residual)

            # Normalization
            phi_temp /= np.max(phi_temp)

            # Calculate errors
            errkeff = np.abs((keff - k_old) / k_old)
            errflux = np.max(np.abs(phi_temp - phi_temp_old) / (np.abs(phi_temp) + 1E-20))

            iter_count += 1
            print(f'Iteration: {iter_count}, keff = {keff:.5f}, errkeff = {errkeff:.6e}, '
                  f'errflux = {errflux:.5e}, residual = {residual_norm:.5e}')

        return keff, phi_temp

class FixedSourceSolver3DHexx:
    def __init__(self, group, conv_tri, M, dS, PHI, precond, tol):
        self.group = group
        self.M = M
        self.dS = dS
        self.tol = tol
        self.conv_tri = conv_tri
        self.PHI = PHI
        self.precond = precond

    def solve(self):
        M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data), comm=PETSc.COMM_WORLD)
        M_petsc.assemble()

        # PETSc Solver (KSP) and Preconditioner (PC)
        ksp = PETSc.KSP().create()
        ksp.setOperators(M_petsc)
        ksp.setType(PETSc.KSP.Type.GMRES)

        # Preconditioner setup
        pc = ksp.getPC()
        if self.precond == 0:
            print(f'Solving using Sparse Solver')
            pc.setType(PETSc.PC.Type.NONE)
        elif self.precond == 1:
            print(f'Solving using ILU')
            pc.setType(PETSc.PC.Type.ILU)
            print(f'ILU Preconditioner Done')
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            pc.setType(PETSc.PC.Type.LU)
            print(f'LU Preconditioner Done')

        # Solver tolerances
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        dPHI_temp = np.ones(self.group*max(self.conv_tri), dtype=complex)
        errdPHI = 1
        iter = 0

        while errdPHI > self.tol:
            dPHI_tempold = dPHI_temp.copy()

            # Set up RHS
            S = self.dS.dot(self.PHI)

            # PETSc Vectors for RHS and solution
            S_petsc = PETSc.Vec().createWithArray(S)
            dPHI_temp_petsc = PETSc.Vec().createWithArray(dPHI_temp)

            # Solve the linear system using PETSc KSP
            ksp.solve(S_petsc, dPHI_temp_petsc)

            # Get result back into NumPy array
            dPHI_temp = dPHI_temp_petsc.getArray()

            # Calculate errors
            errdPHI = np.max(np.abs(dPHI_temp - dPHI_tempold) / (np.abs(dPHI_temp) + 1E-20))

            iter += 1
            print(f'Iteration: {iter}, errflux = {errdPHI:.6e}')

        return dPHI_temp

#######################################################
class PowerMethodSolver2DRectFreeMF:
    def __init__(self, group, N, conv, D, TOT, SIGS, chi, NUFIS,
                 dx, dy, I_max, J_max, BC, precond=None, tol=1e-10, max_it=1000):
        self.group = group
        self.N = N
        self.conv = conv
        self.D = D
        self.TOT = TOT
        self.SIGS = SIGS
        self.chi = chi
        self.NUFIS = NUFIS
        self.dx, self.dy = dx, dy
        self.I_max, self.J_max = I_max, J_max
        self.BC = BC
        self.precond = precond
        self.tol = tol
        self.max_it = max_it

    def solve(self):
        max_conv = max(self.conv)
        n = self.group * max_conv

        # Helper to create a Python (matrix-free) PETSc matrix
        def create_shell_matrix(ctx):
            A = PETSc.Mat().create()
            A.setSizes([n, n])
            A.setType('python')
            A.setPythonContext(ctx)
            A.setUp()
            return A

        # Create matrix-free operators
        M_ctx = MatrixFreeM2DRect(self.group, self.N, self.conv,
                                  self.dx, self.dy, self.D, self.TOT,
                                  self.SIGS, self.BC, self.I_max, self.J_max)
        F_ctx = MatrixFreeF2DRect(self.group, self.N, self.conv,
                                  self.chi, self.NUFIS)

        M = create_shell_matrix(M_ctx)
        F = create_shell_matrix(F_ctx)

        # Setup KSP for solving M φ = S
        ksp = PETSc.KSP().create()
        ksp.setOperators(M)
        ksp.setType("cg")
        ksp.getPC().setType("jacobi")
        ksp.setTolerances(rtol=1e-10, max_it=5000)

        # Initial guess
        phi = PETSc.Vec().createSeq(n)
        phi.set(1.0)
        keff = 1.0

        errflux = errkeff = 1.0
        iter_count = 0

        while errflux > self.tol or errkeff > self.tol:
            phi_old = phi.copy()
            k_old = keff

            # Compute fission source: S = (1/k_old) F φ_old
            S = phi.duplicate()
            F.mult(phi_old, S)
            S.scale(1.0 / k_old)

            # Solve M φ = S
            phi_new = phi.duplicate()
            ksp.solve(S, phi_new)

            # Convert to NumPy for diagnostics
            phi_arr = phi_new.getArray()
            phi_old_arr = phi_old.getArray()

            # Update keff using fission source ratio
            Fphi_new = phi_new.duplicate()
            F.mult(phi_new, Fphi_new)
            Fphi_old = phi_old.duplicate()
            F.mult(phi_old, Fphi_old)

            num = np.trapz(Fphi_new.getArray(), dx=self.dx*self.dy)
            den = np.trapz(Fphi_old.getArray(), dx=self.dx*self.dy)
            keff = k_old * (num / den)

            # Normalize flux
            phi_arr /= np.max(phi_arr)

            # Errors
            errkeff = abs((keff - k_old) / k_old)
            errflux = np.max(np.abs(phi_arr - phi_old_arr) / (np.abs(phi_old_arr) + 1e-20))

            iter_count += 1
            print(f"Iteration {iter_count:3d}: keff={keff:.6f}, "
                  f"errkeff = {errkeff:.6e}, "
                  f'errflux = {errflux:.6e}')

            phi = phi_new

        return keff, phi.getArray()
