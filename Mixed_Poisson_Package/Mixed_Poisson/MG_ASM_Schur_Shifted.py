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
class MGASMShiftedPoisson:
    def __init__(self, height=pi/40, nlayers=20, horiz_num=80, radius=2, mesh="interval", MH=True, refinement=3):
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
        self.W = RT * DG

        # Test Functions
        self.u, self.p = TrialFunctions(self.W)
        self.v, self.q = TestFunctions(self.W)

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
            # self.f = rg.normal(DG, 1.0, 2.0)
            # self.f = as_vector([self.x*(1-self.x), self.y*(1-self.y)])
            self.f = Function(DG).interpolate(10 * exp(-pow(theta, 2)))
        else:
            raise NotImplementedError("this option is not available. Please choose from regular, stiff or random.")
        One = Function(DG).assign(1.0)
        area = assemble(One*dx)
        f_int = assemble(self.f*dx)
        self.f.interpolate(self.f - f_int/area)

    def build_direct_params(self):
        self.params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}

    def build_shifted_params(self):
        self.params = {
                        # 'ksp_type': 'preonly',
                        'ksp_monitor': None,
                        'snes_monitor': None,
                        'snes_type':'ksponly',
                        'ksp_atol': 0,
                        'ksp_rtol': 1e-8,
                        'pc_type':'lu', 
                        'mat_type': 'aij',
                        'pc_factor_mat_solver_type': 'mumps',
                        }


    def build_FieldSplit_params(self):
        helmholtz_schur_pc_params = {
            # 'ksp_type': 'preonly',
            'ksp_monitor': None,
            # 'snes_monitor': None,
            'snes_type':'ksponly',
            # 'ksp_atol': 0,
            # 'ksp_rtol': 1e-9,
            "pc_type": "python",
            # "pc_python_type": "firedrake.ASMStarPC",
            # "pc_star_construct_dim": 0,
            # "pc_star_sub_sub_pc_type": "lu",
            "pc_python_type": "firedrake.ASMVankaPC",
            "pc_vanka_construct_dim": 0,
            "pc_vanka_sub_sub_pc_type": "lu",
            # "pc_vanka_sub_sub_pc_factor_mat_solver_type":'mumps',
        }
        self.params = {
            'ksp_type': 'gmres',
            'snes_type':'ksponly',
            # 'ksp_view': None,
            # 'snes_monitor': None,
            'ksp_monitor': None,
            # 'ksp_atol': 0,
            # 'ksp_rtol': 1e-8,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'full',
            'pc_fieldsplit_0_fields': '1',
            'pc_fieldsplit_1_fields': '0',
            'fieldsplit_0': {
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

    def build_MH_params(self):
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
                            'pc_type': 'python',
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
        self.params = {
            'mat_type': 'aij',
            'ksp_type': 'gmres',
            'snes_type':'ksponly',
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

    def build_NonlinearVariationalSolver(self):
        # Variational Problem
        u = self.u_sol
        p = self.p_sol
        q = self.q
        v = self.v
        f = self.f
        self.F = (inner(u, v) - div(v)*p + div(u)*q)*dx + f * q * dx
        self.shift = (inner(u, v) - div(v)*p + div(u)*q + p * q)*dx # + f * q * dx
        Jp = derivative(self.shift, self.sol)

        # Boundary conditions
        bc1 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "top")
        bc2 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "bottom")
        bc3 = DirichletBC(self.W.sub(0), as_vector([0., 0.]), "on_boundary")
        self.bcs = [bc1, bc2, bc3]

        v_basis = VectorSpaceBasis(constant=True) #pressure field nullspace
        self.nullspace = MixedVectorSpaceBasis(self.W, [self.W.sub(0), v_basis])
        trans_null = VectorSpaceBasis(constant=True)
        self.trans_nullspace = MixedVectorSpaceBasis(self.W, [self.W.sub(0), trans_null])

        self.prob_w = NonlinearVariationalProblem(self.F, self.sol, bcs=self.bcs, Jp=Jp)
        self.solver_w = NonlinearVariationalSolver(self.prob_w,
                                                    nullspace=self.nullspace,
                                                    transpose_nullspace=self.trans_nullspace,
                                                    solver_parameters=self.params, 
                                                    options_prefix='mixed_nonlinear')

    def solve(self, monitor=False, xtest=False, ztest=False, artest=False):

        if monitor:
            self.sol_final = np.loadtxt(f'sol_final.out')
            error_list = []
            # Set a monitor
            def my_monitor_func(ksp, iteration_number, norm):
                #print(f"The monitor is operating with current iteration {iteration_number}")
                sol = ksp.buildSolution()
                # Used relative error here
                err = np.linalg.norm(self.sol_final - sol.getArray(), ord=2) / np.linalg.norm(self.sol_final)
                #print(f"error norm is {err}")
                error_list.append(err)
            self.solver_w.snes.ksp.setMonitor(my_monitor_func)
            self.solver_w.solve()
            # print(error_list)
            print("Monitor is on and working.")
            if artest:
                # test for the aspect ratio
                np.savetxt(f'err_ar_{self.ar}.out', error_list)
            if xtest:
                # test for the different dx
                np.savetxt(f'err_dx_{self.dx}.out', error_list)
            if ztest:
                # test for the different dz
                np.savetxt(f'err_dz_{self.dz}.out', error_list)
        else:
            self.solver_w.solve()
            self.sol_final = self.solver_w.snes.ksp.getSolution().getArray()
            np.savetxt(f'sol_final.out',self.sol_final)

    def write(self):
        sol_u, sol_p = self.sol.subfunctions
        sol_file = VTKFile('sol_MH.pvd')
        sol_file.write(sol_u, sol_p)



if __name__ == "__main__":
        horiz_num = 80
        height = pi / 20
        nlayers = 20
        radius = 2
        refinement = 3
        mesh = "circle"
        option = "random"

        equ = MGASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh, MH=True, refinement=refinement)
        print(f"The calculation is down in a {equ.m.name} mesh.")
        equ.build_f(option=option)
        # equ.build_ASM_MH_params()
        # equ.build_shifted_params()
        # equ.build_FieldSplit_params()
        equ.build_MH_params()
        # equ.build_direct_params()
        equ.build_NonlinearVariationalSolver()
        equ.solve()
        print("!!!!!!!!!!!!!!!",norm(assemble(equ.F, bcs=equ.bcs).riesz_representation()))
        equ.write()

