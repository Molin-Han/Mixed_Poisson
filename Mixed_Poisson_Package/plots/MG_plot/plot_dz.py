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
