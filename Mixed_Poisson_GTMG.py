from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

# Extruded Mesh
m = CircleManifoldMesh(80, radius=2)
height = 4 * pi
nlayers = 20

mesh = ExtrudedMesh(m, nlayers,
                        layer_height = height/nlayers,
                        extrusion_type='radial')

# Create a ExtrudedMesh Hierarchy
mh = MeshHierarchy(m, 0)
hierarchy = ExtrudedMeshHierarchy(mh, height, base_layer=1, refinement_ratio=2, extrusion_type='radial')

# Mixed Finite Element Space
CG_1 = FiniteElement("CG", interval, 1)
DG_0 = FiniteElement("DG", interval, 0)
P1P0 = TensorProductElement(CG_1, DG_0)
RT_horiz = HDivElement(P1P0)
P0P1 = TensorProductElement(DG_0, CG_1)
RT_vert = HDivElement(P0P1)
RT_e = RT_horiz + RT_vert
RT = FunctionSpace(mesh, RT_e)
DG = FunctionSpace(mesh, 'DG', 0)
W = RT * DG

# Test Functions
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
x, y = SpatialCoordinate(mesh)

# Some known function f
theta = atan2(y,x)
f =  Function(DG).interpolate(10 * exp(-pow(theta, 2)))
One =  Function(DG).assign(1.0)
area =  assemble(One*dx)
f_int =  assemble(f*dx)
f.interpolate(f - f_int/area)

# Variational Problem
a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
L = - f * v *  dx

sol = Function(W) # solution in mixed space

# Boundary conditions
bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
bcs = [bc1, bc2]
nullspace = VectorSpaceBasis(constant=True)

params = {
        'mat_type': 'matfree',
        'ksp_type': 'gmres',
        'ksp_monitor': None,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.GTMGPC',
        'gt': {
                'mg_levels': {
                        'ksp_type': 'richardson',
                        # "ksp_converged_reason": None,
                        # "ksp_monitor_true_residual": None,
                        # "ksp_view": None,
                        # "ksp_atol": 1e-8,
                        # "ksp_rtol": 1e-8,
                        "ksp_max_it": 1,
                        'pc_type': 'python',
                        "pc_python_type": "firedrake.AssembledPC",
                        "assembled_pc_type": "python",
                        "assembled_pc_python_type": "firedrake.ASMVankaPC",
                        "assembled_pc_vanka_construct_dim": 0,
                        "assembled_pc_vanka_sub_sub_pc_type": "lu",
                        "assembled_pc_vanka_sub_sub_pc_factor_mat_solver_type":'mumps'
                        },
                'mg_coarse': {
                        'ksp_type': 'preonly',
                        'pc_type': 'lu'
                        }
                }
        }

def get_function_space():
        # TODO: this is not working.
        CG_1 = FiniteElement("CG", interval, 1)
        DG_0 = FiniteElement("DG", interval, 0)
        P1P0 = TensorProductElement(CG_1, DG_0)
        RT_horiz = HDivElement(P1P0)
        P0P1 = TensorProductElement(DG_0, CG_1)
        RT_vert = HDivElement(P0P1)
        RT_e = RT_horiz + RT_vert
        RT = FunctionSpace(mesh, RT_e)
        R = FunctionSpace(mesh, 'R', 0)
        return FunctionSpace(mesh, "RT", 1, vfamily="R", vdegree=0)
def get_coarse_nullspace():
        return VectorSpaceBasis(constant=True)
def function_callback(): # TODO: Check this!
        fspace = get_function_space()
        p = TrialFunction(fspace)
        q = TestFunction(fspace)
        return inner(grad(p), grad(q))*dx
def get_coarse_nullspace():
        return VectorSpaceBasis(constant=True)

# Application Context
appctx = {"get_coarse_operator": function_callback,
            "get_coarse_space": get_function_space,
            "get_coarse_nullspace": get_coarse_nullspace}


# Set the Solver.
prob_w = LinearVariationalProblem(a, L, sol, bcs=bcs)
solver_w = LinearVariationalSolver(prob_w, nullspace=nullspace, solver_parameters=params, appctx=appctx)

solver_w.solve()

sol_u, sol_p = sol.subfunctions

sol_file = VTKFile('sol_GTMG.pvd')
sol_file.write(sol_u, sol_p)
