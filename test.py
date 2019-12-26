import olim
import meshio
import numpy as np

mesh = meshio.read('sphere3.vtk')

points = mesh.points.astype(np.float64)
print('|points| = %d' % (points.shape[0],))

tetra = mesh.cells['tetra'].astype(np.int64)
print('|tetra| = %d' % (tetra.shape[0],))

S = lambda x: 1 + np.dot(x, [0.2, -0.3, 0.1])
slowness = S(points)

tol = np.finfo(np.float64).resolution
solver = olim.EikonalAdaptiveGaussSeidel(points, tetra, slowness, tol)

i0 = np.argmin(np.sum(points**2, axis=1))
print('|points[i0]| = %g' % np.linalg.norm(points[i0]))

solver.add_boundary_point(i0, 0.0)
solver.commit()
solver.solve()