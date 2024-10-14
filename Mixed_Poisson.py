import firedrake as fd
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

# Extruded Mesh
m = fd.CircleManifoldMesh(20, radius=2)
mesh = fd.ExtrudedMesh(m, 5, extrusion_type='radial')

# Mixed Finite Element Space
CG_1 = fd.FiniteElement("CG", fd.interval, 1)
DG_0 = fd.FiniteElement("DG", fd.interval, 0)
P1P0 = fd.TensorProductElement(CG_1, DG_0)
RT_horiz = fd.HDivElement(P1P0)
P0P1 = fd.TensorProductElement(DG_0, CG_1)
RT_vert = fd.HDivElement(P0P1)
RT = RT_horiz + RT_vert
# FIXME: DG still need these operations?
horiz_elt = fd.FiniteElement("DG", fd.triangle, 0)
vert_elt = fd.FiniteElement("DG", fd.interval, 0)
elt = fd.TensorProductElement(horiz_elt, vert_elt)
DG = fd.FunctionSpace(mesh, elt)

W = RT * DG

# Test Functions
sigma, u = fd.TrialFunction(W)
tau, v = fd.TestFunctions(W)

x, y = fd.SpatialCoordinate(mesh)





