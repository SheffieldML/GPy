"""
The file for utilities related to integration by quadrature methods
- will contain implementation for gaussian-kronrod integration.

"""
import numpy as np


def getSubs(Subs, XK, NK=1):
    M = (Subs[1, :] - Subs[0, :]) / 2
    C = (Subs[1, :] + Subs[0, :]) / 2
    G = XK[:, None] * M + np.ones((NK, 1)) * C
    # A = [Subs(1,:); G]
    A = np.vstack((Subs[0, :], G))
    # B = [I;Subs(2,:)]
    B = np.vstack((G, Subs[1, :]))
    # Subs = [reshape(A, 1, []);
    A = A.flatten()
    # reshape(B, 1, [])];
    B = B.flatten()
    Subs = np.vstack((A, B))
    # Subs = np.concatenate((A, B), axis=0)
    return Subs


def quadvgk(feval, fmin, fmax, tol1=1e-5, tol2=1e-5):
    """
    numpy implementation makes use of the code here: http://se.mathworks.com/matlabcentral/fileexchange/18801-quadvgk
    We here use gaussian kronrod integration already used in gpstuff for evaluating one dimensional integrals.
    This is vectorised quadrature which means that several functions can be evaluated at the same time over a grid of
    points.
    :param f:
    :param fmin:
    :param fmax:
    :param difftol:
    :return:
    """

    XK = np.array(
        [
            -0.991455371120813,
            -0.949107912342759,
            -0.864864423359769,
            -0.741531185599394,
            -0.586087235467691,
            -0.405845151377397,
            -0.207784955007898,
            0.0,
            0.207784955007898,
            0.405845151377397,
            0.586087235467691,
            0.741531185599394,
            0.864864423359769,
            0.949107912342759,
            0.991455371120813,
        ]
    )
    WK = np.array(
        [
            0.022935322010529,
            0.063092092629979,
            0.104790010322250,
            0.140653259715525,
            0.169004726639267,
            0.190350578064785,
            0.204432940075298,
            0.209482141084728,
            0.204432940075298,
            0.190350578064785,
            0.169004726639267,
            0.140653259715525,
            0.104790010322250,
            0.063092092629979,
            0.022935322010529,
        ]
    )
    # 7-point Gaussian weightings
    WG = np.array(
        [
            0.129484966168870,
            0.279705391489277,
            0.381830050505119,
            0.417959183673469,
            0.381830050505119,
            0.279705391489277,
            0.129484966168870,
        ]
    )

    NK = WK.size
    # G = np.arange(2, NK, 2)
    tol1 = 1e-4
    tol2 = 1e-4
    Subs = np.array([[fmin], [fmax]])
    #  number of functions to evaluate in the feval vector of functions.
    NF = feval(np.zeros(1)).size
    Q = np.zeros(NF)
    while Subs.size > 0:
        Subs = getSubs(Subs, XK)
        M = (Subs[1, :] - Subs[0, :]) / 2
        C = (Subs[1, :] + Subs[0, :]) / 2
        # NM = length(M);
        NM = M.size
        # x = reshape(XK * M + ones(NK, 1) * C, 1, []);
        x = XK[:, None] * M + C
        x = x.flatten()
        FV = feval(x)
        # FV = FV[:,None]
        Q1 = np.zeros((NF, NM))
        Q2 = np.zeros((NF, NM))

        # for n=1:NF
        # F = reshape(FV(n,:), NK, []);
        # Q1(n,:) = M. * sum((WK * ones(1, NM)). * F);
        # Q2(n,:) = M. * sum((WG * ones(1, NM)). * F(G,:));
        # end
        # for i in range(NF):
        #     F = FV
        #     F = F.reshape((NK,-1))
        #     temp_mat = np.sum(np.multiply(WK[:,None]*np.ones((1,NM)), F),axis=0)
        #     Q1[i,:] = np.multiply(M, temp_mat)
        #     temp_mat = np.sum(np.multiply(WG[:,None]*np.ones((1, NM)), F[G-1,:]), axis=0)
        #     Q2[i,:] = np.multiply(M, temp_mat)
        # ind = np.where(np.logical_or(np.max(np.abs(Q1 -Q2) / Q1) < tol1, (Subs[1,:] - Subs[0,:]) <= tol2) > 0)[0]
        # Q = Q + np.sum(Q1[:,ind], axis=1)
        # np.delete(Subs, ind,axis=1)

        Q1 = np.dot(FV.reshape(NF, NK, NM).swapaxes(2, 1), WK) * M
        Q2 = np.dot(FV.reshape(NF, NK, NM).swapaxes(2, 1)[:, :, 1::2], WG) * M
        # ind = np.nonzero(np.logical_or(np.max(np.abs((Q1-Q2)/Q1), 0) < difftol , M < xtol))[0]
        ind = np.nonzero(
            np.logical_or(
                np.max(np.abs((Q1 - Q2)), 0) < tol1, (Subs[1, :] - Subs[0, :]) < tol2
            )
        )[0]
        Q = Q + np.sum(Q1[:, ind], axis=1)
        Subs = np.delete(Subs, ind, axis=1)
    return Q


def quadgk_int(f, fmin=-np.inf, fmax=np.inf, difftol=0.1):
    """
    Integrate f from fmin to fmax,
    do integration by substitution
    x = r / (1-r**2)
    when r goes from -1 to 1 , x goes from -inf to inf.
    the interval for quadgk function is from -1 to +1, so we transform the space from (-inf,inf) to (-1,1)
    :param f:
    :param fmin:
    :param fmax:
    :param difftol:
    :return:
    """
    difftol = 1e-4

    def trans_func(r):
        r2 = np.square(r)
        x = r / (1 - r2)
        dx_dr = (1 + r2) / (1 - r2) ** 2
        return f(x) * dx_dr

    integrand = quadvgk(trans_func, -1.0, 1.0, difftol, difftol)
    return integrand
