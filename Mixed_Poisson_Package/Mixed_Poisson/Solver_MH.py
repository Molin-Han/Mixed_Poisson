from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson.Poisson import Poisson


class MH_Monitor(Poisson):
        def __init__(self, height=pi/40, nlayers=20, horiz_num=80, radius=2):
                super().__init__(height= height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
                # Create a ExtrudedMesh Hierarchy to achieve the vertical lumping space
                self.mh = MeshHierarchy(self.m, refinement_levels=0)
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

        def solve(self, monitor=False, xtest=False, ztest=False, artest=False):
                '''
                monitor : set a monitor function to trace the error in each iteration
                xtest : store the error for different x number
                ztest : store the error for different z number
                artest : store the error for different aspect ratio
                '''
                if monitor:
                        self.sol_final = np.loadtxt(f'sol_final.out')
                        error_list = []

                        # Set a monitor
                        def my_monitor_func(ksp, iteration_number, norm):
                                #print(f"The monitor is operating with current iteration {iteration_number}")
                                sol = ksp.buildSolution()
                                # TODO: Use relative error here
                                err = np.linalg.norm(self.sol_final - sol.getArray(), ord=2) / np.linalg.norm(self.sol_final)
                                #print(f"error norm is {err}")
                                error_list.append(err)

                        self.solver_w.snes.ksp.setMonitor(my_monitor_func)
                        self.solver_w.solve()
                        #print(f"Solution error list is {error_list}")
                        if artest:
                                # test for the aspect ratio
                                np.savetxt(f'err_ar_{self.ar}.out', error_list)
                        if xtest:
                                # test for the different dx
                                np.savetxt(f'err_dx_{self.dx}.out', error_list)
                        if ztest:
                                # test for the different dz
                                np.savetxt(f'err_dz_{self.dz}.out', error_list)
                else:
                        self.solver_w.solve()

        def write(self):
                sol_u, sol_p = self.sol.subfunctions
                sol_file = VTKFile('sol_MH.pvd')
                sol_file.write(sol_u, sol_p)        

