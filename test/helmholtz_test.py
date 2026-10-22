import sys
import numpy as np
from matplotlib import pyplot as plt
from helmholtz_engine import solverHelmholtz1DFiniteDifferences
from gmri import buildMRI, buildgMRI
from gmri.orthogonalization_engine import orthogonalizationEngine

print(("Test on an FD-discretization of the Helmholtz equation on the interval"
       " [0, pi] with squared wavenumbers between 0 and 10^3. The domain is "
       "discretized using 10^3 grid points. The RHS is random."))
###############################################################################
allowed_tags = ["MRI", "gMRI"]
if len(sys.argv) > 1:
    approximation_strategy = sys.argv[1]
else:
    approximation_strategy = input(("Input approximation_strategy (allowed "
                                    "values: {})\n").format(allowed_tags))
if len(sys.argv) > 2:
    bisections = sys.argv[2]
else:
    bisections = input(("Input the desired number of bisections (the final "
                        "piecewise approximation will be supported on the "
                        "union of 2^bisections subintervals) or 'AUTO':\n"))
    
# If the parameter bisections is not 'AUTO', convert it to an integer
if bisections.upper() != "AUTO": bisections = int(bisections)

if len(sys.argv) > 3:
    eps_stab = float(sys.argv[3])
else:
    eps_stab = float(input("Input the desired stability tolerance:\n"))
###############################################################################
k2s = [0., 1000.]

domain_length = np.pi
n_mesh = 1000
###############################################################################

solver = solverHelmholtz1DFiniteDifferences(domain_length, n_mesh)
energyMatrix = solver.energyMatrix(0.)
engine = orthogonalizationEngine(energyMatrix)

if approximation_strategy == "MRI":
    app = buildMRI(solver.solve, energyMatrix, np.linspace(*k2s, 101),
                   eps_stab, bisections = bisections, is_1d = True)
elif approximation_strategy == "gMRI":
    app = buildgMRI(solver.solve, energyMatrix, np.linspace(*k2s, 10001),
                    1e-3, eps_stab, bisections = bisections, is_1d = True)
#poles = app.poles()
poles = [app_.poles() for app_ in app.apps]

### POSTPROCESS
poles_int_range = [np.ceil(max(1., k2s[0])) ** .5 * (domain_length / np.pi),
                   np.floor(max(1., k2s[-1])) ** .5 * (domain_length / np.pi) + 1]
polesEx = (np.pi * np.arange(*poles_int_range) / domain_length) ** 2

k2test = np.linspace(*k2s, 1001)
u_app = app(k2test, None)
if isinstance(u_app, list):
    normApp = np.empty(len(k2test))
    for j, u_ in enumerate(u_app):
        normApp[j] = np.linalg.norm(u_)
else:
    normApp = np.linalg.norm(u_app, axis = 0)

k2stride = 25
k2testCoarse = k2test[::k2stride]
normEx = np.empty(len(k2testCoarse))
for j, k2 in enumerate(k2testCoarse):
    normEx[j] = engine.norm(solver.solve(k2))[0]

### PLOTS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5)) # horizontal layout
ax1.semilogy(k2testCoarse, normEx, 'o')
ax1.semilogy(k2test, normApp)
for app_ in app.apps:
    ax1.semilogy(app_.support, [np.min(normApp)] * app_.nsupport, 'x')
ax1.legend(["Exact", "Approx"])
ax1.set_xlabel("k"), ax1.set_ylabel("||u(k)||"), ax1.grid()
ax1.set_title("Solution norm")

symbols = "x+.1234"
ax2.plot(polesEx, [0] * len(polesEx), 'o')
for j, poles_ in enumerate(poles):
    ax2.plot(np.real(poles_), np.imag(poles_), symbols[j % len(symbols)])
ax2.legend(["Exact"])
ax2.set_xlabel("Re(poles)"), ax2.set_ylabel("Im(poles)")
ax2.set_xlim(k2s), ax2.set_ylim(-.5, .5), ax2.grid()
ax2.set_title("Poles")

fig.tight_layout()
plt.show()