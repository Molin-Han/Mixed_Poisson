import firedrake as fd
from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson.Poisson import Poisson
# This file solves the Poisson equation without using vertical lumping idea, instead, the pure ASMVanka line smoother is used in each GMRES iteration.

class PureVanka(Poisson):

    def __init__(self, height=fd.pi / 40, nlayers=20, horiz_num=80, radius=2, mesh="interval"):
        super().__init__(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)

    def build_params(self):
        self.params = {
            "mat_type": "matfree",
            "ksp_type": "gmres",
            'snes_monitor': None,
            "ksp_converged_reason": None,
            "snes_converged_reason": None,
            # "ksp_monitor_true_residual": None,
            # "ksp_view": None,
            "ksp_atol": 1e-8,
            "ksp_rtol": 1e-8,
            "ksp_max_it": 400,
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "python",
            "assembled_pc_python_type": "firedrake.ASMVankaPC",
            "assembled_pc_vanka_construct_dim": 0,
            "assembled_pc_vanka_sub_sub_pc_type": "lu",
            "assembled_pc_vanka_sub_sub_pc_factor_mat_solver_type":'mumps'
            }

    def solve(self):
        self.solver_w.solve()

    def write(self):
        sol_u, sol_p = self.sol.subfunctions
        sol_file = VTKFile('sol_pure_Vanka.pvd')
        sol_file.write(sol_u, sol_p)

if __name__ == "__main__":

    horiz_num = 80
    height = fd.pi / 400
    nlayers = 20
    radius = 2
    mesh = "circle"
    option = "regular"

    equ = PureVanka(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ.build_f(option=option)
    equ.build_params()
    equ.build_NonlinearVariationalSolver()
    equ.solve()
    print("!!!!!!!!!!!!!!!",norm(assemble(equ.F, bcs=equ.bcs).riesz_representation()))
    # equ.write()