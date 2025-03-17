from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson import MG_ASM_Schur_Shifted

rate = 8000
height_array = np.exp(np.arange(7, -3, -1.0)) * pi / rate
# height_array = np.exp(np.array([-3.0])) * pi / rate
#height_array = np.array([1.0]) * pi /40
# horiz_num = 80
# nlayers = 20
horiz_num = 100
nlayers = 200
radius = 2
mesh = "circle"
fig, ax = plt.subplots()
ax.set_title(f"The solution error with radius {radius}.")

ar_list = []

for i in height_array:
    print(i)
    height = i
    ar = height/ (2 * pi * radius)
    print(f"Aspect ratio is {ar}")
    ar_list.append(ar)
    equ_MH = MG_ASM_Schur_Shifted.MGASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    print(f"!!!!The calculation is down in a {equ_MH.m.name} mesh.")
    equ_MH.build_f()
    # equ_MH.build_FieldSplit_params()
    equ_MH.build_MH_params()
    equ_MH.build_NonlinearVariationalSolver()
    equ_MH.solve(monitor=False)

    equ_monitor = MG_ASM_Schur_Shifted.MGASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ_monitor.build_f()
    # equ_monitor.build_FieldSplit_params()
    equ_monitor.build_MH_params()
    equ_monitor.build_NonlinearVariationalSolver()
    equ_monitor.solve(monitor=True, artest=True)

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for ar = {ar}")
