from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from Mixed_Poisson_Code import Mixed_Poisson_PureVanka, Mixed_Poisson_MH



height_array = np.arange(10, 6.0, -1.0) * pi / 20
horiz_num = 80
nlayers = 20
radius = 2
#ar_array = height_array / (2 * pi * radius)
ar_array = np.array([]) * pi /40

fig,ax = plt.subplots()
ax.set_title("The solution error")

for ratio in ar_array:
    error = np.loadtxt(f'err_{ratio}.out')
    x = np.arange(len(error))
    ax.loglog(x, error, label=f"AR={ratio}")
    plt.legend()
    plt.xlabel("its")
    plt.ylabel("log_error")
    
plt.savefig("error_final_1.png")