from firedrake import *
# m = UnitIntervalMesh(80)
# mesh = ExtrudedMesh(m, 20, layer_height = pi/400, extrusion_type='uniform')
m = CircleManifoldMesh(80, radius=2, name='circle')
mesh = ExtrudedMesh(m, 20, layer_height = pi/400, extrusion_type='radial')

CG_1 = FiniteElement("CG", interval, 1)
DG_0 = FiniteElement("DG", interval, 0)
P1P0 = TensorProductElement(CG_1, DG_0)
RT_horiz = HDivElement(P1P0)
P0P1 = TensorProductElement(DG_0, CG_1)
RT_vert = HDivElement(P0P1)
RT_e = RT_horiz + RT_vert
RT = FunctionSpace(mesh, RT_e)
DG = FunctionSpace(mesh, 'DG', 0)
# RT = FunctionSpace(mesh, 'RT', 1)
W = RT * DG
tau, v = TestFunctions(W)
sol = Function(W)
sig, u = split(sol)
x, y = SpatialCoordinate(mesh)

f = Function(DG).interpolate(10 * exp(-pow(atan2(y,x), 2)))
area = assemble(Function(DG).assign(1.0)*dx)
f_int = assemble(f*dx)
f.interpolate(f - f_int/area)

F = (inner(sig, tau) + div(tau)*u + div(sig)*v)*dx + f * v * dx

bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0.]), "on_boundary")
bcs = [bc1, bc2, bc3]

# v_basis = VectorSpaceBasis(constant=True) #pressure field nullspace
# nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])

nullspace_v = Function(W)
nullspace_v.sub(1).assign(1 / DG.dim())
nullspace = VectorSpaceBasis([nullspace_v])

trans_null = VectorSpaceBasis(constant=True)
trans_nullspace = MixedVectorSpaceBasis(W, [W.sub(0), trans_null])

params = {'ksp_type': 'gmres','snes_monitor': None,'snes_type':'ksponly', 'ksp_monitor': None,'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
prob_w = NonlinearVariationalProblem(F, sol, bcs=bcs)
solver_w = NonlinearVariationalSolver(prob_w, nullspace=nullspace, transpose_nullspace=trans_nullspace,solver_parameters=params, options_prefix='mixed_nonlinear')

solver_w.solve()
print("Nonlinear Residual is",norm(assemble(F, bcs=bcs).riesz_representation()))