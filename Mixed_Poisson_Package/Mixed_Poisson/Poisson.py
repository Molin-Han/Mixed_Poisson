from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

class Poisson:
    def __init__(self, height=pi/40, nlayers=20, horiz_num=80, radius=2):
        self.height = height
        self.rad = radius
        self.ar = height/(2 * pi * radius)
        self.dx = 2 * pi * radius / horiz_num
        self.dz = height / nlayers
        print(f"The aspect ratio is {self.ar}")

        # self.m = CircleManifoldMesh(horiz_num, radius=radius, name='circle')
        self.m = UnitIntervalMesh(horiz_num, name='interval')
        # Extruded Mesh
        # self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='radial')
        self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='uniform')

        # Mixed Finite Element Space
        CG_1 = FiniteElement("CG", interval, 1)
        DG_0 = FiniteElement("DG", interval, 0)
        P1P0 = TensorProductElement(CG_1, DG_0)
        RT_horiz = HDivElement(P1P0)
        P0P1 = TensorProductElement(DG_0, CG_1)
        RT_vert = HDivElement(P0P1)
        RT_e = RT_horiz + RT_vert
        RT = FunctionSpace(self.mesh, RT_e)
        DG = FunctionSpace(self.mesh, 'DG', 0)
        self.W = RT * DG

        # Test Functions
        self.sigma, self.u = TrialFunctions(self.W)
        self.tau, self.v = TestFunctions(self.W)

        # Solution Functions
        self.sol = Function(self.W) # solution in mixed space
        self.sig_sol, self.u_sol = split(self.sol) #TODO: this is for the NonlinearVariationalSolver.

        self.x, self.y = SpatialCoordinate(self.mesh)

    # TODO: test with stiffer f.
    def build_f(self):
        # Some known function f
        DG = FunctionSpace(self.mesh, 'DG', 0)
        theta = atan2(self.y,self.x)
        # self.f = Function(DG).interpolate(10 * exp(-pow(theta, 2) * (self.y -self.rad) / self.height))
        self.f = Function(DG).interpolate(exp(-pow(theta, 2)*self.y))
        #self.f = Function(DG).interpolate(exp(-pow(theta, 2)))
        One = Function(DG).assign(1.0)
        area = assemble(One*dx)
        f_int = assemble(self.f*dx)
        self.f.interpolate(self.f - f_int/area)
    
    def build_params(self):
        self.params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}

    def build_LinearVariationalSolver(self):
        # Variational Problem
        self.a = (dot(self.sigma, self.tau) + div(self.tau)*self.u + div(self.sigma)*self.v)*dx
        self.L = - self.f * self.v * dx

        # Boundary conditions
        bc1 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "top")
        bc2 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "bottom")
        self.bcs = [bc1, bc2]

        self.nullspace = VectorSpaceBasis(constant=True)
        self.prob_w = LinearVariationalProblem(self.a, self.L, self.sol, bcs=self.bcs)
        self.solver_w = LinearVariationalSolver(self.prob_w, nullspace=self.nullspace, solver_parameters=self.params)

    def build_NonlinearVariationalSolver(self):
        # Variational Problem
        self.F = (dot(self.sig_sol, self.tau) + div(self.tau)*self.u_sol + div(self.sig_sol)*self.v)*dx + self.f * self.v * dx

        # Boundary conditions
        bc1 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "top")
        bc2 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "bottom")
        self.bcs = [bc1, bc2]

        self.nullspace = VectorSpaceBasis(constant=True)
        self.prob_w = NonlinearVariationalProblem(self.F, self.sol, bcs=self.bcs)
        self.solver_w = NonlinearVariationalSolver(self.prob_w, nullspace=self.nullspace, solver_parameters=self.params)

