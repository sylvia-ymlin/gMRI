import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from gmri import buildMRI, buildgMRI
from gmri.orthogonalization_engine import orthogonalizationEngine

from matplotlib.patches import Rectangle

print(("Test on the 'gun' nonlinear eigenvalue problem from the NLEVP "
       "collection mantained by F. Tisseur."))
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
if len(sys.argv) > 3:
    eps_stab = float(sys.argv[3])
else:
    eps_stab = float(input("Input the desired stability tolerance:\n"))
###############################################################################
k2s, alphas = [1.25e4, 1.125e5], [0., 108.8774]
# data = loadmat('nlevp/private/gun.mat')

import os 
# load data from where this script is located
data = loadmat(os.path.join(os.path.dirname(__file__), 'nlevp/private/gun.mat'))

data_mat = {key: csc_matrix(data[key]) for key in ["K", "M", "W1", "W2"]}
np.random.seed(42)
rhs = np.random.randn(data_mat["K"].shape[0])
lhs = lambda k2: (data_mat["K"] - k2 * data_mat["M"]
                + 1j * ((k2 - alphas[0] ** 2) ** .5 * data_mat["W1"]
                      + (k2 - alphas[1] ** 2) ** .5 * data_mat["W2"]))
###############################################################################

solve = lambda k2: spsolve(lhs(k2), rhs).reshape(-1, 1)
energyMatrix = data_mat["M"]
engine = orthogonalizationEngine(energyMatrix)

if approximation_strategy == "MRI":
    app = buildMRI(solve, energyMatrix, np.linspace(*k2s, 101),
                   eps_stab, bisections = bisections, is_1d = True)
elif approximation_strategy == "gMRI":
    app = buildgMRI(solve, energyMatrix, np.linspace(*k2s, 10001),
                    1e-3, eps_stab, bisections = bisections, is_1d = True)
#poles = app.poles()
poles = [app_.poles() for app_ in app.apps] # only for coloring

### POSTPROCESS
k2test = np.linspace(*k2s, 1001)
u_app = app(k2test, None)
if isinstance(u_app, list):
    normApp = np.empty(len(k2test))
    for j, u_ in enumerate(u_app):
        normApp[j] = np.linalg.norm(u_)
else:
    normApp = np.linalg.norm(u_app, axis = 0)

k2stride = 40
k2testCoarse = k2test[::k2stride]
normEx = np.empty(len(k2testCoarse))
for j, k2 in enumerate(k2testCoarse):
    normEx[j] = engine.norm(solve(k2))[0]

### PLOTS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5)) # horizontal layout
ax1.semilogy(k2testCoarse, normEx, 'o')
ax1.semilogy(k2test, normApp)
for app_ in app.apps:
    ax1.semilogy(app_.support, [np.min(normApp)] * app_.nsupport, 'x')
ax1.legend(["Exact", "Approx"])
ax1.set_xlabel("k"), ax1.set_ylabel("||u(k)||"), ax1.grid()
ax1.set_xlabel("k"), ax1.set_ylabel("||u(k)||"), ax1.grid()
ax1.set_title("Solution norm")

symbols = "x+.1234"

delta_rel = 0.05

for j, (app_, p) in enumerate(zip(app.apps, poles)):
    c = f"C{j}"
    a, b = np.min(np.real(app_.support)), np.max(np.real(app_.support))
    d = delta_rel * (b - a)
    # 可信区域：Re ∈ [a, b], |Im| ≤ d
    ax2.add_patch(Rectangle((a, -d), b - a, 2 * d, fc=c, ec=c, alpha=.15))
    ok = (p.real >= a) & (p.real <= b) & (np.abs(p.imag) <= d)
    ax2.plot(p[ok].real,  p[ok].imag,  symbols[j % len(symbols)], color=c)
    ax2.plot(p[~ok].real, p[~ok].imag, symbols[j % len(symbols)], color=c, alpha=.25)

# 分支割线 (-∞, α₂²] 和分支点 0, α₂²
xl = ax2.get_xlim()
bp = [alphas[0] ** 2, alphas[1] ** 2]
ax2.plot([xl[0], bp[1]], [0, 0], 'k-', lw=4, alpha=.35, zorder=0, label="branch cut")
ax2.plot(bp, [0, 0], 'k*', ms=10, label="branch points")
ax2.set_xlim(xl)          # 画割线后恢复横轴范围，避免被拉宽

ax2.legend(loc="upper right")
ax2.set_xlabel("Re(poles)"), ax2.set_ylabel("Im(poles)"), ax2.grid()
ax2.set_title("Poles (faded: outside trusted region)")

fig.tight_layout()
plt.show()

