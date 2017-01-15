import numpy


def read(data_file):
    return numpy.loadtxt(data_file)


def create_affinity(coordinates):
    points_num = len(coordinates[0])  # number of points
    W = numpy.zeros((points_num, points_num,))
    for i in xrange(points_num):
        for j in range(i + 1, points_num):
            diff = numpy.array((coordinates[0][i], coordinates[1][i])) - \
                   numpy.array((coordinates[0][j], coordinates[1][j]))
            W[i][j] = numpy.linalg.norm(diff)
            W[j][i] = W[i][j]
    W = apply_gaussian(W)
    return W


def apply_gaussian(W):
    return W