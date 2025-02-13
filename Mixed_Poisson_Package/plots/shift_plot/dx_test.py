from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson import shifted_Poisson as Schur_Shifted

height = pi / 40
horiz_array = np.arange(2, 50, 5) * 2
nlayers = 20
radius = 2
mesh = "circle"
option = "random"
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

    equ_MH = Schur_Shifted.ShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ_MH.build_f(option=option)
    # equ_MH.build_FieldSplit_params()
    equ_MH.build_shifted_params()
    equ_MH.build_NonlinearVariationalSolver()
    equ_MH.solve(monitor=False)

    equ_monitor = Schur_Shifted.ShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ_monitor.build_f(option=option)
    # equ_monitor.build_FieldSplit_params()
    equ_monitor.build_shifted_params()
    equ_monitor.build_NonlinearVariationalSolver()
    equ_monitor.solve(monitor=True, xtest=True)

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for dx = {dx}")
i = 0
for dx in dx_list:
    horiz = horiz_array[i]
    i += 1
    error = np.loadtxt(f'err_dx_{dx}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"horiz={horiz}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{dx}.png")

plt.savefig(f"error_final_dx_{option}_{dx}.png")
