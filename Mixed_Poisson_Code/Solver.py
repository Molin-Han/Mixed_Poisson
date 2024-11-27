from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile


class MH_Monitor:
# TODO: need to rethink this monitor option.
        def __init__(self, height=pi/40, nlayers=20, horiz_num=80, radius=2):
                super().__init__(height= height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)

        def build_f(self):
                # Some known function f
                DG = FunctionSpace(self.mesh, 'DG', 0)
                theta = atan2(self.y,self.x)
                self.f = Function(DG).interpolate(10 * exp(-pow(theta, 2)))
                One = Function(DG).assign(1.0)
                area = assemble(One*dx)
                f_int = assemble(self.f*dx)
                self.f.interpolate(self.f - f_int/area)

        def build_LinearVariationalSolver(self):
                # Variational Problem
                self.a = (dot(self.sigma, self.tau) + div(self.tau)*self.u + div(self.sigma)*self.v)*dx
                self.L = - self.f * self.v * dx
                self.sol = Function(self.W) # solution in mixed space

                # Boundary conditions
                bc1 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "top")
                bc2 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "bottom")
                self.bcs = [bc1, bc2]
                self.nullspace = VectorSpaceBasis(constant=True)

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
                self.prob_w = LinearVariationalProblem(self.a, self.L, self.sol, bcs=self.bcs)
                self.solver_w = LinearVariationalSolver(self.prob_w, nullspace=self.nullspace, solver_parameters=self.params)

        def solve(self):
                if self.monitor == True:
                    self.sol_final = np.loadtxt(f'sol_final_{self.ar}.out')
                    #print(f"Our solution array is of length {len(self.sol_final)}, and is a {type(self.sol_final)}")
                    error_list = []

                    # Set a monitor
                    def my_monitor_func(ksp, iteration_number, norm):
                            #print(f"The monitor is operating with current iteration {iteration_number}")
                            sol = ksp.buildSolution()
                            # if iteration_number < 10 and iteration_number > 5 :
                            #         np.savetxt(f'sol_{self.ar}_{iteration_number}.out', sol)
                            # print(f"The solution at current step is {sol.getArray()}")
                            err = np.linalg.norm(self.sol_final - sol.getArray(), ord=2)
                            #print(f"error norm is {err}")
                            error_list.append(err)

                    self.solver_w.snes.ksp.setMonitor(my_monitor_func)
                    self.solver_w.solve()
                    #print(f"Solution error list is {error_list}")
                    # test for the aspect ratio
                    #np.savetxt(f'err_{self.ar}.out', error_list)
                    # test for the different dx
                    #np.savetxt(f'err_dx_{self.dx}.out', error_list)
                    # test for the different dz
                    np.savetxt(f'err_dz_{self.dz}.out', error_list)
                else:
                    self.solver_w.solve()


        def write(self):
                sol_u, sol_p = self.sol.subfunctions
                sol_file = VTKFile('sol_MH.pvd')
                sol_file.write(sol_u, sol_p)        

