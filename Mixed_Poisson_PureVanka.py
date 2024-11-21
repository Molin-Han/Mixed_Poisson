import firedrake as fd
from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

# Extruded Mesh
m = fd.CircleManifoldMesh(80, radius=2)
height = fd.pi / 40
nlayers = 20
mesh = fd.ExtrudedMesh(m, nlayers,
                       layer_height = height/nlayers,
                       extrusion_type='radial')

# Mixed Finite Element Space
CG_1 = fd.FiniteElement("CG", fd.interval, 1)
DG_0 = fd.FiniteElement("DG", fd.interval, 0)
P1P0 = fd.TensorProductElement(CG_1, DG_0)
RT_horiz = fd.HDivElement(P1P0)
P0P1 = fd.TensorProductElement(DG_0, CG_1)
RT_vert = fd.HDivElement(P0P1)
RT_e = RT_horiz + RT_vert
RT = fd.FunctionSpace(mesh, RT_e)
DG = fd.FunctionSpace(mesh, 'DG', 0)
W = RT * DG

# Test Functions
sigma, u = fd.TrialFunctions(W)
tau, v = fd.TestFunctions(W)

x, y = fd.SpatialCoordinate(mesh)

# Some known function f
theta = fd.atan2(y,x)
f = fd.Function(DG).interpolate(10 * fd.exp(-pow(theta, 2)))
One = fd.Function(DG).assign(1.0)
area = fd.assemble(One*fd.dx)
f_int = fd.assemble(f*fd.dx)
f.interpolate(f - f_int/area)

# Variational Problem
a = (fd.dot(sigma, tau) + fd.div(tau)*u + fd.div(sigma)*v)*fd.dx
L = - f * v * fd.dx
sol = fd.Function(W) # solution in mixed space

# Boundary conditions
bc1 = fd.DirichletBC(W.sub(0), fd.as_vector([0., 0.]), "top")
bc2 = fd.DirichletBC(W.sub(0), fd.as_vector([0., 0.]), "bottom")
bcs = [bc1, bc2]
nullspace = fd.VectorSpaceBasis(constant=True)

params = {
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_monitor_true_residual": None,
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

prob_w = fd.LinearVariationalProblem(a, L, sol, bcs=bcs)
solver_w = fd.LinearVariationalSolver(prob_w, nullspace=nullspace, solver_parameters=params)

solver_w.solve()
sol_final = solver_w.snes.ksp.getSolution().getArray()
np.savetxt('sol_final.out',sol_final)
print(len(sol_final), type(sol_final))
sol_u, sol_p = sol.subfunctions

sol_file = VTKFile('sol_pure_Vanka.pvd')
sol_file.write(sol_u, sol_p)
