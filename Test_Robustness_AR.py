from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson_Code import Mixed_Poisson_PureVanka, Mixed_Poisson_MH

rate = 60000
height_array = np.arange(6, 2.0, -1.0) * pi / rate
#height_array = np.array([1.0]) * pi /40
horiz_num = 80
nlayers = 20
radius = 2
fig, ax = plt.subplots()
ax.set_title("The solution error")


ar_list = []
for i in height_array:
    print(i)
    height = i
    ar = height/ (2 * pi * radius)
    print(f"Aspect ratio is {ar}")
    ar_list.append(ar)
    equ_PV = Mixed_Poisson_PureVanka.PureVanka(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
    equ_PV.build_f()
    equ_PV.build_LinearVariationalSolver()
    equ_PV.solve()

    equ_MH = Mixed_Poisson_MH.MH(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius)
    equ_MH.build_f()
    equ_MH.build_LinearVariationalSolver()
    equ_MH.solve()

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for ar = {ar}")
    # error = np.loadtxt(f'err_{ar}.out')
    # x = np.arange(len(error))
    # ax.semilogy(x, error, label=f"AR={ar}")
    # plt.legend()
    # plt.xlabel("its")
    # plt.ylabel("log_error")
    # plt.savefig(f"error{ar}.png")



for ratio in ar_list:
    error = np.loadtxt(f'err_{ratio}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"AR={ratio}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    plt.savefig(f"error_final{ratio}.png")
    
plt.savefig(f"error_final_ar_{ratio}.png")
