from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson import MG_ASM_Schur_Shifted


horiz_array = np.arange(2, 40, 10) * 5
# horiz_array = np.array([32]) * 5
# height = pi / 40
# nlayers = 20
height = pi / 2000
nlayers = 100
radius = 2
mesh = "interval"
fig, ax = plt.subplots()
ax.set_title("The solution error for different dx")

dx_list = []

for i in horiz_array:
    print(i)
    horiz_num = i
    dx = 2 * pi * radius / horiz_num
    ar = height/ (2 * pi * radius)
    print(f"Aspect ratio is {ar}")
    print(f"The dx is {dx}")
    dx_list.append(dx)

    equ_MH = MG_ASM_Schur_Shifted.MGASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ_MH.build_f()
    # equ_MH.build_FieldSplit_params()
    # equ_MH.build_shifted_params()
    equ_MH.build_MH_params()
    equ_MH.build_NonlinearVariationalSolver()
    equ_MH.solve(monitor=False)

    equ_monitor = MG_ASM_Schur_Shifted.MGASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ_monitor.build_f()
    # equ_monitor.build_FieldSplit_params()
    # equ_monitor.build_shifted_params()
    equ_monitor.build_MH_params()
    equ_monitor.build_NonlinearVariationalSolver()
    equ_monitor.solve(monitor=True, xtest=True)

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for dx = {dx}")
