import sys
import data_reader
import numpy
import scipy.linalg as la
import random
import matplotlib.pyplot as plt


def diagonalize(w):
    ones_vector = numpy.ones(w.shape[0])
    return numpy.diag(numpy.dot(w, ones_vector))


def find_eigensolution_z(w, d, k):
    # compute inverse of square root of D
    d_s = la.sqrtm(d)
    d_s_i = numpy.linalg.inv(d_s)
    t = numpy.dot(numpy.dot(d_s_i, w), d_s_i)
    v, s, u_t = numpy.linalg.svd(t)
    idxs = s.argsort()[::-1] # get indexes for sorting eigenvalues and eigenvectors
    s = s[idxs]  # sort eigenvalues
    v = v[:, idxs]  # sort eigenvectors
    v = v[:, :k]
    s = numpy.diag(s)[:k, :k]
    return numpy.dot(d_s_i, v)


def normalize(z):
    m = numpy.diag(numpy.dot(z, z.transpose()))
    m = numpy.diag(m)
    m = numpy.linalg.inv(la.sqrtm(m))
    x = numpy.dot(m, z)
    return x


def initialize_r(x):
    n = x.shape[0]  # number of rows
    k = x.shape[1]  # number of clusters
    i = random.randint(0, n - 1)
    r = numpy.zeros([k, k])
    r[:, 0] = x[i, :].transpose()
    c = numpy.zeros(n)
    for j in xrange(1, k):
        c += numpy.absolute(numpy.dot(x, r[:, j - 1]))
        i = numpy.argmin(c)
        r[:, j] = x[i, :].transpose()
    return r


def find_discrete_x(x_tl, r):
    x_tl = numpy.dot(x_tl, r)
    n = x_tl.shape[0]
    k = x_tl.shape[1]
    x = numpy.empty([n, k])
    for i in xrange(n):
        for j in xrange(k):
            x[i, j] = 1 if j == numpy.argmax(x_tl[i, :]) else 0
    return x


def process_clustering(dataset_file, clusters_num, epsilon):
    points = data_reader.read(dataset_file)
    # W = affinity matrix
    w = data_reader.create_affinity(points)
    # D = Diag(W * I)
    d = diagonalize(w)
    # matrix to decompose
    z = find_eigensolution_z(w, d, clusters_num)
    # normalized z
    x_tl = normalize(z)
    # initialize r
    r = initialize_r(x_tl)
    # initialize convergence monitoring factor
    theta = 0.
    x_disc = None
    while True:
        x_disc = find_discrete_x(x_tl, r)
        to_decomp = numpy.dot(x_disc.transpose(), x_tl)
        u, omega, u_s_t = numpy.linalg.svd(to_decomp)
        theta_s = numpy.sum(omega)
        if abs(theta_s - theta) < epsilon:
            break
        theta = theta_s
        r = numpy.dot(u_s_t.transpose(), u.transpose())
    return x_disc, points


def main(argv):
    dataset_file = argv[1]
    clusters_num = int(argv[2])
    epsilon = float(argv[3])
    x, points = process_clustering(dataset_file, clusters_num, epsilon)
    for i in xrange(clusters_num):
        print numpy.sum(x[:, i])
    numpy.savetxt('data/res.txt', x)
    n = len(points[0])
    colors = ['ro', 'go', 'bo', 'yo']
    for i in xrange(clusters_num):
        x_coord = []
        y_coord = []
        for j in xrange(x.shape[0]):
            if x[j][i] == 1:
                x_coord.append(points[0][j])
                y_coord.append(points[1][j])
        plt.plot(x_coord, y_coord, colors[i % 4])
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
