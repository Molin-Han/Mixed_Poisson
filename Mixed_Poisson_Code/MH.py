from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile


class MH:
        def __init__(self, height=pi/40, nlayers=20, horiz_num=80, radius=2):
                self.ar = height/(2 * pi * radius)
                self.dx = 2 * pi * radius / horiz_num
                self.dz = height / nlayers
                print(f"The aspect ratio is {self.ar}")
                m = CircleManifoldMesh(horiz_num, radius=radius)
                # Extruded Mesh
                self.mesh = ExtrudedMesh(m, nlayers,
                                layer_height = height/nlayers,
                                extrusion_type='radial')

                # Create a ExtrudedMesh Hierarchy to achieve the vertical lumping space
                self.mh = MeshHierarchy(m, refinement_levels=0)
                self.hierarchy = ExtrudedMeshHierarchy(self.mh, height,layers=[1, nlayers], extrusion_type='radial')
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

                self.x, self.y = SpatialCoordinate(self.mesh)

        def build_f(self):
                # Some known function f
                DG = FunctionSpace(self.mesh, 'DG', 0)
                theta = atan2(self.y,self.x)
                self.f = Function(DG).interpolate(10 * exp(-pow(theta, 2)))
                One = Function(DG).assign(1.0)
                area = assemble(One*dx)
                f_int = assemble(self.f*dx)
                self.f.interpolate(self.f - f_int/area)

        def build_LinearVariationalSolver(self):
                # Variational Problem
                self.a = (dot(self.sigma, self.tau) + div(self.tau)*self.u + div(self.sigma)*self.v)*dx
                self.L = - self.f * self.v * dx
                self.sol = Function(self.W) # solution in mixed space

                # Boundary conditions
                bc1 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "top")
                bc2 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "bottom")
                self.bcs = [bc1, bc2]
                self.nullspace = VectorSpaceBasis(constant=True)

                self.params = {
                        'mat_type': 'matfree',
                        'ksp_type': 'gmres',
                        #'ksp_monitor': None,
                        "ksp_monitor_true_residual": None,
                        'pc_type': 'mg',
                        'pc_mg_type': 'full',
                        'mg_levels': {
                                'ksp_type': 'richardson',
                                # "ksp_converged_reason": None,
                                # "ksp_monitor_true_residual": None,
                                # "ksp_view": None,
                                # "ksp_atol": 1e-8,
                                # "ksp_rtol": 1e-8,
                                'ksp_max_it': 1,
                                'pc_type': 'python',
                                'pc_python_type': 'firedrake.AssembledPC',
                                'assembled_pc_type': 'python',
                                'assembled_pc_python_type': 'firedrake.ASMVankaPC',
                                'assembled_pc_vanka_construct_dim': 0,
                                'assembled_pc_vanka_sub_sub_pc_type': 'lu'
                                #'assembled_pc_vanka_sub_sub_pc_factor_mat_solver_type':'mumps'
                                },
                        'mg_coarse': {
                                'ksp_type': 'preonly',
                                'pc_type': 'lu'
                                }
                        }
                self.prob_w = LinearVariationalProblem(self.a, self.L, self.sol, bcs=self.bcs)
                self.solver_w = LinearVariationalSolver(self.prob_w, nullspace=self.nullspace, solver_parameters=self.params)

        #TODO: No need to set monitor here.
        def solve(self):
                self.solver_w.solve()


        def write(self):
                sol_u, sol_p = self.sol.subfunctions
                sol_file = VTKFile('sol_MH.pvd')
                sol_file.write(sol_u, sol_p)        



if __name__ == "__main__":

        horiz_num = 80
        height = pi / 400
        nlayers = 20
        radius = 2

        equ = MH(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
        equ.build_f()
        equ.build_LinearVariationalSolver()
        equ.solve()
        # equ.write()
