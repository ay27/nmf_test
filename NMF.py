import numpy


def nmf_gd(R, P, Q, K, steps=1000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        print('step = %d' %step)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e += pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    # for k in range(K):
                    #     e += (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        print('e = %f' % e)
    return P, Q.T


def nmf_mul(W, H, V, steps=50000, r=2):
    n = len(V)
    m = len(V[0])
    for step in range(steps):
        WV = numpy.dot(W.T, V)
        WWH = numpy.dot(W.T, numpy.dot(W, H))
        for a in range(r):
            for u in range(m):
                H[a, u] = H[a, u] * WV[a, u] / WWH[a, u]

        VH = numpy.dot(V, H.T)
        WHH = numpy.dot(W, numpy.dot(H, H.T))
        for i in range(n):
            for a in range(r):
                W[i, a] = W[i, a] * VH[i, a] / WHH[i, a]

        WH = numpy.dot(W, H)
        e = 0
        for i in range(len(V)):
            for j in range(len(V[i])):
                if V[i][j] > 0:
                    e += pow(V[i][j] - WH[i][j], 2)
        if e < 0.001:
            print('e = %f\n' % e)
            break

    return W, H.T
