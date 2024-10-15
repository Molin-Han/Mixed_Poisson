import firedrake as fd
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

# Extruded Mesh
m = fd.CircleManifoldMesh(20, radius=2)

height = 0.1
nlayers = 5

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

# FIXME: DG still need these operations?
horiz_elt = fd.FiniteElement("DG", fd.interval, 0)
vert_elt = fd.FiniteElement("DG", fd.interval, 0)
elt = fd.TensorProductElement(horiz_elt, vert_elt)
DG = fd.FunctionSpace(mesh, elt)

W = RT * DG

# Test Functions
print(len(W))
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
# f = fd.Function(DG).interpolate(
#     10*fd.exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

a = (fd.dot(sigma, tau) + fd.div(tau)*u + fd.div(sigma)*v)*fd.dx
L = - f * v * fd.dx

sol = fd.Function(W) # solution in mixed space

# TODO: Need to impose the Boundary condition: use null space to do this?


bc1 = fd.DirichletBC(W.sub(0), fd.as_vector([0., 0.]), "top")
bcs = [bc1]
bc1 = fd.DirichletBC(W.sub(0), fd.as_vector([0., 0.]), "bottom")
bcs.append(bc1)
nullspace = fd.VectorSpaceBasis(constant=True)


# TODO: Set up a solver

params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}

prob_w = fd.LinearVariationalProblem(a, L, sol, bcs=bcs)
solver_w = fd.LinearVariationalSolver(prob_w, nullspace=nullspace, solver_parameters=params)


solver_w.solve()

sol_u, sol_p = sol.subfunctions

sol_file = VTKFile('sol.pvd')
sol_file.write(sol_u, sol_p)

