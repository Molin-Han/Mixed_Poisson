from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson.Poisson import Poisson
# This code implements the MeshHierarchy for solving the Mixed Poisson equation that enables the vertical lumping idea to create the vertically constant space as the coarse grid. And solving the fine grid with an ASMVanka PC. This file codes the Mixed Poisson equation with NonlinearVariationalSolver.

class PoissonMeshHierarchy(Poisson):
        def __init__(self, height=pi/40, nlayers=20, horiz_num=80, radius=2, mesh="interval"):
                super().__init__(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
                # Create a ExtrudedMesh Hierarchy to achieve the vertical lumping space
                self.mh = MeshHierarchy(self.m, refinement_levels=0)
                if mesh == "interval":
                        self.hierarchy = ExtrudedMeshHierarchy(self.mh, height,layers=[1, nlayers], extrusion_type='uniform')
                if mesh == "circle":
                        self.hierarchy = ExtrudedMeshHierarchy(self.mh, height,layers=[1, nlayers], extrusion_type='radial')


        def build_params(self):
                self.params = {
                        'mat_type': 'matfree',
                        'ksp_type': 'gmres',
                        'snes_monitor': None,
                        # 'snes_type':'ksponly',
                        'ksp_monitor': None,
                        # "ksp_monitor_true_residual": None,
                        'pc_type': 'mg',
                        'pc_mg_type': 'full',
                        "ksp_converged_reason": None,
                        "snes_converged_reason": None,
                        'mg_levels': {
                                'ksp_type': 'richardson',
                                # "ksp_monitor_true_residual": None,
                                # "ksp_view": None,
                                "ksp_atol": 1e-50,
                                "ksp_rtol": 1e-10,
                                'ksp_max_it': 1,
                                'pc_type': 'python',
                                'pc_python_type': 'firedrake.AssembledPC',
                                'assembled_pc_type': 'python',
                                'assembled_pc_python_type': 'firedrake.ASMVankaPC',
                                'assembled_pc_vanka_construct_dim': 0,
                                'assembled_pc_vanka_sub_sub_pc_type': 'lu'
                                #'assembled_pc_vanka_sub_sub_pc_factor_mat_solver_type':'mumps'
                                },
                        'mg_coarse': {
                                'ksp_type': 'preonly',
                                'pc_type': 'lu'
                                }
                        }

        def params_direct(self):
                self.params = {'ksp_type': 'gmres','snes_monitor': None,
                                # 'snes_type':'ksponly', 
                                "snes_converged_reason": None,
                                'ksp_monitor': None,'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}

        def solve(self, monitor=False):
                self.solver_w.solve()
                if monitor:
                        self.sol_final = self.solver_w.snes.ksp.getSolution().getArray()
                        np.savetxt(f'sol_final.out',self.sol_final)

        def write(self):
                sol_u, sol_p = self.sol.subfunctions
                sol_file = VTKFile('sol_MH.pvd')
                sol_file.write(sol_u, sol_p)        


if __name__ == "__main__":
        horiz_num = 80
        height = pi / 20
        nlayers = 20
        radius = 2
        mesh = "interval"
        option = "regular"

        equ = PoissonMeshHierarchy(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
        print(f"The calculation is down in a {equ.m.name} mesh.")
        equ.build_f(option=option)
        equ.build_params()
        # equ.params_direct()
        # equ.build_LinearVariationalSolver()
        equ.build_NonlinearVariationalSolver()
        equ.solve()
        print("!!!!!!!!!!!!!!!",norm(assemble(equ.F, bcs=equ.bcs).riesz_representation()))
        equ.write()
