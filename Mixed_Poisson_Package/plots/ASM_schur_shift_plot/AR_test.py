from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson import ASM_Schur_shifted

rate = 8000
height_array = np.arange(15, 6.0, -1.0) * pi / rate
#height_array = np.array([1.0]) * pi /40
# horiz_num = 80
# nlayers = 20
horiz_num = 100
nlayers = 200
radius = 2
mesh = "circle"
option = "random"
fig, ax = plt.subplots()
ax.set_title(f"The solution error with radius {radius}.")

ar_list = []

for i in height_array:
    print(i)
    height = i
    ar = height/ (2 * pi * radius)
    print(f"Aspect ratio is {ar}")
    ar_list.append(ar)
    equ_MH = ASM_Schur_shifted.ASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    print(f"!!!!The calculation is down in a {equ_MH.m.name} mesh.")
    equ_MH.build_f(option=option)
    # equ_MH.build_shifted_params()
    equ_MH.build_FieldSplit_params()
    equ_MH.build_NonlinearVariationalSolver()
    equ_MH.solve(monitor=False)

    equ_monitor = ASM_Schur_shifted.ASMShiftedPoisson(height=height, nlayers=nlayers, horiz_num=horiz_num, radius=radius, mesh=mesh)
    equ_monitor.build_f(option=option)
    # equ_monitor.build_shifted_params()
    equ_monitor.build_FieldSplit_params()
    equ_monitor.build_NonlinearVariationalSolver()
    equ_monitor.solve(monitor=True, artest=True)

    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!Finish Calculation for ar = {ar}")

j = 0
for ratio in ar_list:
    error = np.loadtxt(f'err_ar_{ratio}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"ar={round(ratio,5)}")
    j+=1
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{ratio}.png")
    
plt.savefig(f"ar_{option}_{ratio}.png")
