from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
# This is the implementation without using the Auxiliary Operator PC to express Jp for the shifted PC and use Schur complement to eliminate the pressure. 
# This file also implement the multigrid to the fieldsplit0 Hdiv velocity block to ensure the robustness in horizontal mesh size.
# This is a script without class object to improve the readability of the code.
class HDivHelmholtzSchurPC(AuxiliaryOperatorPC):
    _prefix = "helmholtzschurpc_"
    def form(self, pc, u, v):
        W = u.function_space()
        Jp = (inner(u, v) + div(v)*div(u))*dx
        #  Boundary conditions
        bc1 = DirichletBC(W, as_vector([0., 0.]), "top")
        bc2 = DirichletBC(W, as_vector([0., 0.]), "bottom")
        bc3 = DirichletBC(W, as_vector([0., 0.]), "on_boundary")
        bcs = [bc1, bc2, bc3]
        return (Jp, bcs)

height=pi/40
nlayers=20
horiz_num=80
refinement=3

m = UnitIntervalMesh(horiz_num, name='interval')
mesh = ExtrudedMesh(m, nlayers, layer_height = height/nlayers, extrusion_type='uniform')
mh = MeshHierarchy(m, refinement_levels=refinement)
hierarchy = ExtrudedMeshHierarchy(mh, height,base_layer=nlayers,refinement_ratio=1, extrusion_type='uniform')
mesh = hierarchy[-1]
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

sol = Function(W) # solution in mixed space
u, p = split(sol)
v, q = TestFunctions(W)
x, y = SpatialCoordinate(mesh)

f_DG = VectorFunctionSpace(mesh, 'DG', 0)
# f = Function(f_DG).interpolate(as_vector([x*(1-x),y*(1-y)]))

theta = atan2(y,x)
# f = Function(DG).interpolate(exp(-pow(theta, 2)*y))

pcg = PCG64(seed=123456789)
rg = Generator(pcg)
f = rg.normal(f_DG, 1.0, 2.0)

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
                            "pc_type": "python",
                            # "pc_python_type": "firedrake.ASMStarPC", # TODO: shall we use AssembledPC?
                            # "pc_star_construct_dim": 0,
                            # "pc_star_sub_sub_pc_type": "lu",
                            "pc_python_type": "firedrake.ASMVankaPC", # TODO: shall we use AssembledPC?
                            "pc_vanka_construct_dim": 0,
                            "pc_vanka_sub_sub_pc_type": "lu",
                        },
            'mg_coarse': {'ksp_type': 'preonly',
                            'pc_type': 'lu',
                    },
}
params = {
            'mat_type': 'aij',
            'ksp_type': 'gmres',
            'snes_type':'ksponly',
            'ksp_atol': 0,
            'ksp_rtol': 1e-8,
            # 'ksp_view': None,
            'snes_monitor': None,
            # 'ksp_monitor': None,
            'ksp_monitor_true_residual': None,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'full',
            'pc_fieldsplit_0_fields': '1',
            'pc_fieldsplit_1_fields': '0',
            'fieldsplit_0': { # Doing a pure mass solve for the pressure block.
                'ksp_type': 'preonly',
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu',
                # 'pc_factor_mat_solver_type': 'mumps',
            },
            'fieldsplit_1': {
                'ksp_type': 'preonly',
                'pc_type': 'python',
                'pc_python_type': __name__ + '.HDivHelmholtzSchurPC',
                'helmholtzschurpc': helmholtz_schur_pc_params,
                }
}

F = (inner(u, v) - div(v)*p + div(u)*q)*dx
F += - inner(f,v) * dx
# F += f * q * dx
shift = (inner(u, v) - div(v)*p + div(u)*q + p * q)*dx # + f * q * dx
Jp = derivative(shift, sol)

bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0.]), "on_boundary")
bcs = [bc1, bc2, bc3]

v_basis = VectorSpaceBasis(constant=True) #pressure field nullspace
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])
trans_null = VectorSpaceBasis(constant=True)
trans_nullspace = MixedVectorSpaceBasis(W, [W.sub(0), trans_null])

prob_w = NonlinearVariationalProblem(F, sol, bcs=bcs, Jp=Jp)
solver_w = NonlinearVariationalSolver(prob_w,nullspace=nullspace,transpose_nullspace=trans_nullspace,solver_parameters=params)
solver_w.solve()
