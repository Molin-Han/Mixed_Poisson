from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
# This file solves individually the Hdiv velocity block which will appear in the Mixed Poisson system using a naive preconditioner solving with multigrid and a ASMStar patch line smoother to improve horizonal robustness and solving efficiency.
# This file is implemented in the way that no class object is included to reduce the complexness for code interpretation and understanding.
class HDivHelmholtzSchurPC(AuxiliaryOperatorPC):
    _prefix = "helmholtzschurpc_"
    def form(self, pc, u, v):
        W = u.function_space()
        Jp = (inner(u, v) + div(v)*div(u))*dx
        #  Boundary conditions
        bcs = DirichletBC(W, 0., "on_boundary")
        return (Jp, bcs)

horiz_num = 80
height = 1
nlayers = 20
refinement = 2

mesh = UnitSquareMesh(horiz_num, nlayers)
mh = MeshHierarchy(mesh, refinement_levels=refinement)
mesh = mh[-1]

W = FunctionSpace(mesh, 'RT', 1)
v = TestFunction(W)
u = Function(W)
x, y = SpatialCoordinate(mesh)

pcg = PCG64(seed=123456789)
rg = Generator(pcg)
f = rg.normal(W, 1.0, 2.0)
# f = as_vector([x*(1-x), y*(1-y)])

helmholtz_schur_pc_params = {
    'ksp_type': 'preonly',
    'ksp_max_its': 30,
    'pc_type': 'mg',
    'pc_mg_type': 'full',
    'pc_mg_cycle_type':'v',
    'mg_levels': {
                    # 'ksp_type': 'gmres',
                    'ksp_type':'richardson',
                    'ksp_richardson_scale': 1/4,
                    'ksp_max_it': 1,
                    # 'ksp_monitor':None,
                    'pc_type': 'python',
                    "pc_python_type": "firedrake.ASMStarPC", # TODO: shall we use AssembledPC?
                    "pc_star_construct_dim": 0,
                    "pc_star_sub_sub_pc_type": "lu",
                    # "pc_python_type": "firedrake.ASMVankaPC", # TODO: shall we use AssembledPC?
                    # "pc_vanka_construct_dim": 0,
                    # "pc_vanka_sub_sub_pc_type": "lu",
                },
    'mg_coarse': {'ksp_type': 'preonly',
                    'pc_type': 'lu',
                }
}
params = {
    "mat_type": "aij",
    'ksp_type': 'gmres',
    'snes_type':'ksponly',
    # 'ksp_view': None,
    'snes_monitor': None,
    # 'ksp_monitor': None,
    'ksp_monitor_true_residual':None,
    'pc_type': 'python',
    'pc_python_type': __name__ + '.HDivHelmholtzSchurPC',
    'helmholtzschurpc': helmholtz_schur_pc_params,
}

# TODO: This is the parameter that did not use the preconditioner.
# params = {
#     "mat_type": "aij",
#     'ksp_type': 'gmres',
#     'snes_type':'ksponly',
#     # 'ksp_view': None,
#     'snes_monitor': None,
#     # 'ksp_monitor': None,
#     'ksp_monitor_true_residual':None,
#     'pc_type': 'mg',
#     'pc_mg_type': 'full',
#     'pc_mg_cycle_type':'v',
#     'mg_levels': {
#                     'ksp_type': 'gmres',
#                     'ksp_max_it': 1,
#                     # 'ksp_monitor': None,
#                     # 'pc_type': 'lu',
#                     'pc_type': 'python', #TODO: Should we use AssembledPC?
#                     "pc_python_type": "firedrake.ASMStarPC",
#                     "pc_star_construct_dim": 0,
#                     "pc_star_sub_sub_pc_type": "lu",
#                     },
#     'mg_coarse': {'ksp_type': 'preonly',
#                     'pc_type': 'lu',
#                  }
# }

F = inner(u,v)*dx + div(v) * div(u) * dx - inner(f,v) * dx
bcs = DirichletBC(W, 0., "on_boundary")
prob_w = NonlinearVariationalProblem(F, u, bcs=bcs)
solver_w = NonlinearVariationalSolver(prob_w, solver_parameters=params)
solver_w.solve()
