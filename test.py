import meshio
import numpy as np
import python_example as olim

mesh = meshio.read('sphere3.vtk')

points = mesh.points.astype(np.float64)
tetra = mesh.cells['tetra'].astype(np.int64)

S = lambda x: 1 + np.dot(x, [0.2, -0.3, 0.1])

slowness = S(points)

solver = olim.EikonalTetraSolver(points, tetra, slowness)
