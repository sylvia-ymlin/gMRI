"""Given-degree MRI: degree constraint M_N q = 0 and two normalizations.

Euclidean:  min ||R q||  s.t.  M_N q = 0, ||q||_2 = 1         (SVD)
Monic:      min ||R q||  s.t.  M_N q = 0, ell_N q = 1         (least squares)

Moment vectors are built by Vandermonde with Arnoldi on scaled nodes
(Brubeck, Nakatsukasa, Trefethen, SIAM Rev. 63, 2021).
"""
import numpy as np
from .mri import MRI, StabilityError

__all__ = ['constraintBasis', 'givenDegreeMRI']


def constraintBasis(zs, N):
    """
    Orthonormal basis Z of ker M_N (deg Q <= N). For the monic
        normalization it is split as Z = [d, U], with ell_N d != 0 and U a
        basis of ker M_N cap ker ell_N.

    Args:
        zs: support points (vector of length S);
        N: denominator degree, 0 <= N <= S - 1;

    Returns:
        Z: S x (N+1) matrix.
    """
    zs = np.asarray(zs).flatten() # accept row/column vectors
    S = len(zs)
    k = S - N - 1 # number of moment conditions (rows of M_N); dim ker M_N = N + 1

    if k < 0: raise ValueError("Need N <= S - 1.")

    # Vandermonde with Arnoldi gives an orthonormal basis of
    # span{1, x, ..., x^(k-1)} (rows of M_N) and x^k (ell_N); ker M_N is the
    # orthogonal complement of the first part, obtained from a complete QR.

    # scale and shift the support points into [-1, 1] (unit disk for complex
    # nodes) for numerical stability
    center = zs.mean()
    scale = np.max(np.abs(zs - center))
    if scale == 0.: scale = 1.
    x = (zs - center) / scale        # degree constraints are invariant under affine maps

    # Vandermonde with Arnoldi: span V[:, :i+1] = span{1, x, ..., x^i}
    # build the Q factor of the Vandermonde matrix column by column (Krylov
    # space of diag(x) and the constant vector); this avoids the
    # ill-conditioned Vandermonde matrix itself

    V = np.zeros((S, k + 1), dtype = complex) # complex dtype for complex support points
    V[:, 0] = 1. / np.sqrt(S) # p_0 = constant
    for i in range(k):
        v = x * V[:, i] # next column: diag(x) V[:, i], x = scaled support points
        for _ in range(2):           # Gram-Schmidt twice (CGS2): the second pass restores the orthogonality lost to rounding in the first
            v = v - V[:, : i + 1] @ (V[:, : i + 1].conj().T @ v)
        V[:, i + 1] = v / np.linalg.norm(v)

    # M_N q = 0  <=>  V[:, :k]^T q = 0  <=>  q orthogonal to conj(V[:, :k]);
    # column k is the only one with nonzero ell_N, i.e., V[:, k]^T q
    return np.linalg.qr(V.conj(), mode = "complete")[0][:, k :]


class givenDegreeMRI(MRI):
    def __init__(self, sampler, energy_matrix, supp, N, eps_stab = None,
                 starting_sampler_data = None, normalization = "euclidean"):
        """
        Initialize minimal rational interpolant with given denominator degree.

        Args:
            sampler, energy_matrix, supp, eps_stab, starting_sampler_data,
                normalization: as in MRI;
            N: denominator degree, N <= S - 1.
        """
        super().__init__(sampler, energy_matrix, supp, eps_stab,
                         starting_sampler_data, normalization)
        self.N = N

    def build(self, supp):
        """
        Build barycentric form with deg Q <= N (Euclidean) or Q monic of
            degree N (monic). For N = S - 1 there is no degree constraint and
            this is MRI.build.
        """
        supp = np.asarray(supp).flatten()

        if self.N == len(supp) - 1:
            super().build(supp)
            return

        # degree constraint: M_N q = 0, with normalization
        R = self.sampler.Rfactor # R factor of the snapshots [u(z_1), ..., u(z_S)] in the energy inner product
        Z = constraintBasis(supp, self.N) # orthonormal basis of ker M_N

        # Euclidean: min ||R q|| s.t. M_N q = 0, ||q||_2 = 1, SVD
        if self.normalization == "euclidean":
            y, unstable = self._minSingVec(R @ Z)
            coeffs = Z @ y # ||q|| = ||y|| since Z is orthonormal
        else:
            # ell_N q = 1 up to a scalar factor, which does not change the
            # barycentric form: min ||R (d + U x)||
            d, U = Z[:, 0], Z[:, 1 :]
            coeffs = d + U @ np.linalg.lstsq(R @ U, - R @ d, rcond = None)[0] # q = d + U x*, x* = argmin_x ||R (d + U x)||
            unstable = False # no stability check since we don't use SVD here.

        # coeffs = q (denominator weights)
        # R * coeffs: numerator weights, column j = q_j R[:, j] = q_j u(z_j) in the orthonormal basis Q ([u(z_1), ..., u(z_S)] = Q R, Q energy-orthonormal);
        #               .T so that row j belongs to support point z_j
        self.setBarycentric(supp, coeffs, (R * coeffs).T)
        if unstable: raise StabilityError("givenDegreeMRI stability check fail")

    def poles(self, return_residues = False, mult = "snaps_ortho"):
        """
        Poles of the barycentric form. For N < S - 1, the arrowhead
            eigenproblem returns S - 1 roots, S - 1 - N of which are at
            infinity in exact arithmetic but large and finite in floating
            point; keep the N roots closest to the support.
        """
        p = super().poles()
        if self.N < self.nsupport - 1:
            p = p[np.sort(np.argsort(np.abs(p - np.mean(self.support)))[: self.N])]
        if return_residues: return p, self(p, mult, ders = [0, 1]).T
        return p
