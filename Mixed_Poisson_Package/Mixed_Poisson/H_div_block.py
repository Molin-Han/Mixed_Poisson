from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
# TODO: This is the implementation without using the Auxiliary Operator PC to express Jp for the shifted PC and use Schur complement to eliminate the pressure. 
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
class ASMShiftedPoisson:
    def __init__(self, height=pi/40, nlayers=20, horiz_num=80, mesh="interval", MH=True, refinement=3):
        '''
        mesh : interval or circle to be extruded.
        '''
        self.height = height
        # self.rad = radius
        # self.ar = height/(2 * pi * radius)
        # self.dx = 2 * pi * radius / horiz_num
        self.dz = height / nlayers
        # print(f"The aspect ratio is {self.ar}")

        # Extruded Mesh
        if mesh == "interval":
            self.m = UnitIntervalMesh(horiz_num, name='interval')
            self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='uniform')

        if MH:
            # Create a ExtrudedMesh Hierarchy to achieve the vertical lumping space
            self.mh = MeshHierarchy(self.m, refinement_levels=refinement)
            if mesh == "interval":
                self.hierarchy = ExtrudedMeshHierarchy(self.mh, height,base_layer=nlayers,refinement_ratio=1, extrusion_type='uniform')
            if mesh == "circle":
                self.hierarchy = ExtrudedMeshHierarchy(self.mh, height,base_layer=nlayers,refinement_ratio=1, extrusion_type='radial')

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
        self.W = RT

        # Test Functions
        self.v = TestFunction(self.W)
        # Solution Functions
        self.u_sol = Function(self.W) # solution in mixed space

        self.x, self.y = SpatialCoordinate(self.mesh)

    def build_f(self):
        DG = FunctionSpace(self.mesh, 'DG', 0)
        RT = self.W
        theta = atan2(self.y,self.x)

        pcg = PCG64(seed=123456789)
        rg = Generator(pcg)
        self.f = rg.normal(RT, 1.0, 2.0)
        # self.f = Function(DG).interpolate(10 * exp(-pow(theta, 2)))
        # One = Function(DG).assign(1.0)
        # area = assemble(One*dx)
        # f_int = assemble(self.f*dx)
        # self.f.interpolate(self.f - f_int/area)

    def build_FieldSplit_params(self):
        helmholtz_schur_pc_params = {
            'ksp_type': 'preonly',
            'ksp_converged_reason':None,
            'ksp_max_its': 30,
            'pc_type': 'mg',
            'pc_mg_type': 'full',
            'mg_levels': {
                            'ksp_type': 'richardson',
                            'ksp_max_it': 1,
                            'pc_type': 'python',
                            'ksp_monitor':None,
                            "pc_python_type": "firedrake.ASMStarPC",
                            "pc_star_construct_dim": 0,
                            "pc_star_sub_sub_pc_type": "lu",
                            # 'pc_python_type': 'firedrake.AssembledPC',
                            # 'assembled_pc_type': 'python',
                            # 'assembled_pc_python_type': 'firedrake.ASMStarPC',
                            # 'assembled_pc_star_construct_dim': 0,
                            # 'assembled_pc_star_sub_sub_pc_type': 'lu',
                            #'assembled_pc_vanka_sub_sub_pc_factor_mat_solver_type':'mumps'
                            },
            'mg_coarse': {'ksp_type': 'preonly',
                            'pc_type': 'lu',
                            }
            # "pc_type": "python",
            # "pc_python_type": "firedrake.ASMStarPC",
            # "pc_star_construct_dim": 0,
            # "pc_star_sub_sub_pc_type": "lu",
            # "pc_star_sub_sub_pc_factor_mat_solver_type":'mumps',
        }
        self.params = {
            'ksp_type': 'fgmres',
            # 'ksp_type': 'preonly',
            'snes_type':'ksponly',
            'ksp_view': None,
            'snes_monitor': None,
            # 'ksp_monitor': None,
            'ksp_monitor_true_residual':None,
            # 'pc_type': 'python',
            # 'pc_python_type': __name__ + '.HDivHelmholtzSchurPC',
            # 'helmholtzschurpc': helmholtz_schur_pc_params,
            'pc_type': 'mg',
            'pc_mg_type': 'full',
            'pc_mg_cycle_type':'v',
            'mg_levels': {
                            'ksp_type': 'gmres',
                            'ksp_max_it': 1,
                            'pc_type': 'python',
                            'ksp_monitor':None,
                            "pc_python_type": "firedrake.ASMStarPC",
                            "pc_star_construct_dim": 0,
                            "pc_star_sub_sub_pc_type": "lu",
                            # 'pc_python_type': 'firedrake.AssembledPC',
                            # 'assembled_pc_type': 'python',
                            # 'assembled_pc_python_type': 'firedrake.ASMStarPC',
                            # 'assembled_pc_star_construct_dim': 0,
                            # 'assembled_pc_star_sub_sub_pc_type': 'lu',
                            #'assembled_pc_vanka_sub_sub_pc_factor_mat_solver_type':'mumps'
                            },
            'mg_coarse': {'ksp_type': 'preonly',
                            'pc_type': 'lu',
                            }
        }

    def build_NonlinearVariationalSolver(self):
        # Variational Problem
        u = self.u_sol
        v = self.v
        f = self.f
        self.F = (inner(u,v))*dx + div(v) * div(u) * dx
        self.F += - inner(f,v) * dx

        # Boundary conditions
        bc1 = DirichletBC(self.W, as_vector([0., 0.]), "top")
        bc2 = DirichletBC(self.W, as_vector([0., 0.]), "bottom")
        bc3 = DirichletBC(self.W, as_vector([0., 0.]), "on_boundary")
        self.bcs = [bc1, bc2, bc3]

        self.prob_w = NonlinearVariationalProblem(self.F, self.u_sol, bcs=self.bcs)
        self.solver_w = NonlinearVariationalSolver(self.prob_w,
                                                    solver_parameters=self.params)

    def solve(self):
            self.solver_w.solve()
            self.sol_final = self.solver_w.snes.ksp.getSolution().getArray()
            np.savetxt(f'sol_final.out',self.sol_final)

    def write(self):
        sol_u = self.u_sol
        sol_file = VTKFile('sol_MH.pvd')
        sol_file.write(sol_u)


if __name__ == "__main__":
        horiz_num = 80
        height = 1
        nlayers = 20

        equ = ASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num)
        print(f"The calculation is down in a {equ.m.name} mesh.")
        equ.build_f()
        equ.build_FieldSplit_params()
        equ.build_NonlinearVariationalSolver()
        equ.solve()
        equ.write()
