from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson import ASM_Schur_shifted

# height = pi / 40
# horiz_num = 80
height = pi / 2000
horiz_num = 100
nlayers_array = np.arange(2, 11, 2) * 100
radius = 2
mesh = "circle"
option = "random"
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

    equ_MH = ASM_Schur_shifted.ASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ_MH.build_f(option=option)
    equ_MH.build_FieldSplit_params()
    # equ_MH.build_shifted_params()
    equ_MH.build_NonlinearVariationalSolver()
    equ_MH.solve(monitor=False)

    equ_monitor = ASM_Schur_shifted.ASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ_monitor.build_f(option=option)
    equ_monitor.build_FieldSplit_params()
    # equ_monitor.build_shifted_params()
    equ_monitor.build_NonlinearVariationalSolver()
    equ_monitor.solve(monitor=True, ztest=True)

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for dz = {dz}")

i = 0
for dz in dz_list:
    nlayer = nlayers_array[i]
    i += 1
    error = np.loadtxt(f'err_dz_{dz}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"nlayer={nlayer}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{dz}.png")

plt.savefig(f"dz_{option}_{dz}.png")
