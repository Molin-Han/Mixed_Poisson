import firedrake as fd
from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile

class PureVanka:

    def __init__(self, height=fd.pi / 40, nlayers=20, horiz_num=80, radius=2):
        self.ar = height/(2 * fd.pi * radius)
        print(f"The aspect ratio is {self.ar}")
        m = fd.CircleManifoldMesh(horiz_num, radius=radius)
        # Extruded Mesh
        self.mesh = fd.ExtrudedMesh(m, nlayers,
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
        RT = fd.FunctionSpace(self.mesh, RT_e)
        DG = fd.FunctionSpace(self.mesh, 'DG', 0)
        self.W = RT * DG

        # Test Functions
        self.sigma, self.u = fd.TrialFunctions(self.W)
        self.tau, self.v = fd.TestFunctions(self.W)

        self.x, self.y = fd.SpatialCoordinate(self.mesh)

    def build_f(self):
        # Some known function f
        DG = fd.FunctionSpace(self.mesh, 'DG', 0)
        theta = fd.atan2(self.y,self.x)
        self.f = fd.Function(DG).interpolate(10 * fd.exp(-pow(theta, 2)))
        One = fd.Function(DG).assign(1.0)
        area = fd.assemble(One*fd.dx)
        f_int = fd.assemble(self.f*fd.dx)
        self.f.interpolate(self.f - f_int/area)

    def build_LinearVariationalSolver(self):
        # Variational Problem
        self.a = (fd.dot(self.sigma, self.tau) + fd.div(self.tau)*self.u + fd.div(self.sigma)*self.v)*fd.dx
        self.L = - self.f * self.v * fd.dx
        self.sol = fd.Function(self.W) # solution in mixed space

        # Boundary conditions
        bc1 = fd.DirichletBC(self.W.sub(0), fd.as_vector([0., 0.]), "top")
        bc2 = fd.DirichletBC(self.W.sub(0), fd.as_vector([0., 0.]), "bottom")
        self.bcs = [bc1, bc2]
        self.nullspace = fd.VectorSpaceBasis(constant=True)

        self.params = {
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

        self.prob_w = fd.LinearVariationalProblem(self.a, self.L, self.sol, bcs=self.bcs)
        self.solver_w = fd.LinearVariationalSolver(self.prob_w, nullspace=self.nullspace, solver_parameters=self.params)

    def solve(self):
        self.solver_w.solve()
        self.sol_final = self.solver_w.snes.ksp.getSolution().getArray()
        np.savetxt(f'sol_final_{self.ar}.out',self.sol_final)
        print(f"Our solution array is of length {len(self.sol_final)}, and is a {type(self.sol_final)}")

    def write(self):
        sol_u, sol_p = self.sol.subfunctions
        sol_file = VTKFile('sol_pure_Vanka.pvd')
        sol_file.write(sol_u, sol_p)

if __name__ == "__main__":

    horiz_num = 80
    height = fd.pi / 40
    nlayers = 20
    radius = 2

    equ = PureVanka(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
    equ.build_f()
    equ.build_LinearVariationalSolver()
    equ.solve()
    # equ.write()