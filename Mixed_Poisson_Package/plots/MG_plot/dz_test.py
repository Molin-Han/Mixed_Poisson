from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson import MG_ASM_Schur_Shifted

# height = pi / 40
# horiz_num = 80
height = pi / 2000
horiz_num = 100
nlayers_array = np.exp(np.arange(2, 11, 2)* 2/5) * 10
nlayers_array = nlayers_array.astype(int)
# nlayers_array = np.array([8.0]) * 100
radius = 2
mesh = "circle"
fig, ax = plt.subplots()
ax.set_title("The solution error for different dz")

dz_list = []

for i in nlayers_array:
    print(i)
    nlayers = i
    dz = height / nlayers
    ar = height/ (2 * pi * radius)
    print(f"Aspect ratio is {ar}")
    print(f"The dz is {dz}")
    dz_list.append(dz)

    equ_MH = MG_ASM_Schur_Shifted.MGASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ_MH.build_f()
    equ_MH.build_MH_params()
    equ_MH.build_NonlinearVariationalSolver()
    equ_MH.solve(monitor=False)

    equ_monitor = MG_ASM_Schur_Shifted.MGASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ_monitor.build_f()
    equ_monitor.build_MH_params()
    equ_monitor.build_NonlinearVariationalSolver()
    equ_monitor.solve(monitor=True, ztest=True)

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for dz = {dz}")
