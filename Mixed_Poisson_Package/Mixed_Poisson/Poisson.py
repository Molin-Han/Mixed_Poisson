from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

class Poisson:
    def __init__(self, height=pi/40, nlayers=20, horiz_num=80, radius=2, mesh="interval"):
        '''
        mesh : interval or circle to be extruded.
        '''
        self.height = height
        self.rad = radius
        self.ar = height/(2 * pi * radius)
        self.dx = 2 * pi * radius / horiz_num
        self.dz = height / nlayers
        print(f"The aspect ratio is {self.ar}")
        
        # Extruded Mesh
        if mesh == "interval":
            self.m = UnitIntervalMesh(horiz_num, name='interval')
            self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='uniform')
        if mesh == "circle":
            self.m = CircleManifoldMesh(horiz_num, radius=radius, name='circle')
            self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='radial')

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
        self.u, self.p = TrialFunctions(self.W)
        self.tau, self.v = TestFunctions(self.W)

        # Solution Functions
        self.sol = Function(self.W) # solution in mixed space
        self.u_sol, self.p_sol = split(self.sol)

        self.x, self.y = SpatialCoordinate(self.mesh)

    def build_f(self, option="regular"):
        '''
        option : regular or stiff or random
        '''
        DG = FunctionSpace(self.mesh, 'DG', 0)
        theta = atan2(self.y,self.x)
        if option == "regular":
            self.f = Function(DG).interpolate(10 * exp(-pow(theta, 2)))

        elif option == "stiff":
            # self.f = Function(DG).interpolate(10 * exp(-pow(theta, 2) * (self.y -self.rad) / self.height))
            self.f = Function(DG).interpolate(exp(-pow(theta, 2)*self.y))
            #self.f = Function(DG).interpolate(exp(-pow(theta, 2)))

        elif option == "random":
            pcg = PCG64(seed=123456789)
            rg = Generator(pcg)
            self.f = rg.normal(DG, 1.0, 2.0)
            # self.f = Function(DG).interpolate(10 * exp(-pow(theta, 2)))
        else:
            raise NotImplementedError("this option is not available. Please choose from regular, stiff or random.")
        One = Function(DG).assign(1.0)
        area = assemble(One*dx)
        f_int = assemble(self.f*dx)
        self.f.interpolate(self.f - f_int/area)

    def build_params(self):
        self.params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}

    def build_LinearVariationalSolver(self):
        # Variational Problem
        self.a = (dot(self.u, self.tau) + div(self.tau)*self.p + div(self.u)*self.v)*dx
        self.L = - self.f * self.v * dx

        # Boundary conditions
        bc1 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "top")
        bc2 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "bottom")
        bc3 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "on_boundary")
        self.bcs = [bc1, bc2, bc3]

        v_basis = VectorSpaceBasis(constant=True) #pressure field nullspace
        self.nullspace = MixedVectorSpaceBasis(self.W, [self.W.sub(0), v_basis])
        trans_null = VectorSpaceBasis(constant=True)
        self.trans_nullspace = MixedVectorSpaceBasis(self.W, [self.W.sub(0), trans_null])

        self.prob_w = LinearVariationalProblem(self.a, self.L, self.sol, bcs=self.bcs)
        self.solver_w = LinearVariationalSolver(self.prob_w, nullspace=self.nullspace,
                                                transpose_nullspace=self.trans_nullspace, 
                                                solver_parameters=self.params, 
                                                options_prefix='mixed_linear')

    def build_NonlinearVariationalSolver(self):
        # Variational Problem
        u = self.u_sol
        p = self.p_sol
        tau = self.tau
        v = self.v
        f = self.f
        self.F = (inner(u, tau) + div(tau)*p + div(u)*v)*dx + f * v * dx

        # Boundary conditions
        bc1 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "top")
        bc2 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "bottom")
        bc3 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "on_boundary")
        self.bcs = [bc1, bc2, bc3]

        v_basis = VectorSpaceBasis(constant=True) #pressure field nullspace
        self.nullspace = MixedVectorSpaceBasis(self.W, [self.W.sub(0), v_basis])
        trans_null = VectorSpaceBasis(constant=True)
        self.trans_nullspace = MixedVectorSpaceBasis(self.W, [self.W.sub(0), trans_null])

        self.prob_w = NonlinearVariationalProblem(self.F, self.sol, bcs=self.bcs)
        self.solver_w = NonlinearVariationalSolver(self.prob_w,
                                                    nullspace=self.nullspace,
                                                    transpose_nullspace=self.trans_nullspace,
                                                    solver_parameters=self.params, 
                                                    options_prefix='mixed_nonlinear')

