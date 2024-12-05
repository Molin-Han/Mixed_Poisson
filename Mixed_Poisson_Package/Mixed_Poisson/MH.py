from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson.Poisson import Poisson


class PoissonMeshHierarchy(Poisson):
        def __init__(self, height=pi/40, nlayers=20, horiz_num=80, radius=2):
                super().__init__(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
                # Create a ExtrudedMesh Hierarchy to achieve the vertical lumping space
                self.mh = MeshHierarchy(self.m, refinement_levels=0)
                # vertical lumping space
                self.hierarchy = ExtrudedMeshHierarchy(self.mh, height,layers=[1, nlayers], extrusion_type='radial')
                # self.hierarchy = ExtrudedMeshHierarchy(self.mh, height,layers=[1, nlayers], extrusion_type='uniform')


        def build_params(self):
                self.params = {
                        'mat_type': 'matfree',
                        'ksp_type': 'gmres',
                        #'ksp_monitor': None,
                        "ksp_monitor_true_residual": None,
                        'pc_type': 'mg',
                        'pc_mg_type': 'full',
                        'mg_levels': {
                                'ksp_type': 'richardson',
                                # "ksp_converged_reason": None,
                                # "ksp_monitor_true_residual": None,
                                # "ksp_view": None,
                                # "ksp_atol": 1e-8,
                                # "ksp_rtol": 1e-8,
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
        height = pi / 400
        nlayers = 20
        radius = 2

        equ = PoissonMeshHierarchy(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
        equ.build_f()
        equ.build_params()
        #equ.build_LinearVariationalSolver()
        equ.build_NonlinearVariationalSolver()
        equ.solve()
        equ.write()
