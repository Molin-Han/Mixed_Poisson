from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson_Code import Mixed_Poisson_PureVanka, Mixed_Poisson_MH

height = pi / 40
horiz_array = np.arange(2, 11, 2) * 50
nlayers = 20
radius = 2
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

    equ_PV = Mixed_Poisson_PureVanka.PureVanka(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
    equ_PV.build_f()
    equ_PV.build_LinearVariationalSolver()
    equ_PV.solve()

    equ_MH = Mixed_Poisson_MH.MH(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
    equ_MH.build_f()
    equ_MH.build_LinearVariationalSolver()
    equ_MH.solve()

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for dx = {dx}")

for dx in dx_list:
    error = np.loadtxt(f'err_dx_{dx}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"dx={dx}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    plt.savefig(f"error_final{dx}.png")

plt.savefig(f"error_final_dx_{dx}.png")
