from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson import MH, Solver_MH

height = pi / 40
horiz_num = 80
nlayers_array = np.arange(2, 11, 2) * 20
radius = 2
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

    equ_MH = MH.PoissonMeshHierarchy(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
    equ_MH.build_f()
    equ_MH.build_params()
    equ_MH.build_LinearVariationalSolver()
    equ_MH.solve(monitor=True)

    equ_monitor = Solver_MH.MH_Monitor(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
    equ_monitor.build_f()
    equ_monitor.build_params()
    equ_monitor.build_LinearVariationalSolver()
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

plt.savefig(f"error_final_dz_{dz}.png")
