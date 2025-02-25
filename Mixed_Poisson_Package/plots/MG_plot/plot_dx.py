from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson import MG_ASM_Schur_Shifted


horiz_array = np.arange(2, 40, 10) * 5
# height = pi / 40
# nlayers = 20
height = pi / 2000
nlayers = 100
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

plt.savefig(f"dx_{option}_{dx}.png")
