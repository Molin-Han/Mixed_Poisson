from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson import MG_ASM_Schur_Shifted

rate = 8000
height_array = np.exp(np.arange(7, -3, -1.0)) * pi / rate
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

j = 0
for ratio in ar_list:
    error = np.loadtxt(f'err_ar_{ratio}.out')
    x = np.arange(len(error))
    ax.semilogy(x, error, label=f"ar={round(ratio,6)}")
    j+=1
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    #plt.savefig(f"error_final{ratio}.png")
    
plt.savefig(f"ar_{option}_{ratio}.png")
