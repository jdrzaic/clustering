import numpy as np
from scipy import ndimage
from scipy import misc
from PIL import Image
import math
import clustering
import data_reader

city_block_radius = 5
k = 3

def test_edge():
    im = ndimage.imread("../data/flower.png", True)

    sx = ndimage.sobel(im, axis=0, mode='constant')
    sy = ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    width, height = sob.shape

    pts = np.array([[float(height - y), float(x)] for x in range(width) for y in range(height)]).transpose()
    affinity = np.full((width * height, width * height,), 0.00000000001);
    for x in range(width):
        for y in range(height):
            for i in range(-city_block_radius, city_block_radius + 1):
                for j in range(-city_block_radius, city_block_radius + 1):
                    if x + i < 0 or x + i >= width or y + j < 0 or y + j >= height or (i == 0 and j == 0): continue
                    max_edge = 0.
                    mag = math.sqrt(math.pow(i, 2) + math.pow(j, 2))
                    for t in np.arange(0., 1., 0.1):
                        e = sob[int(round(x + i * t))][int(round(y + j * t))]
                        if e > max_edge: max_edge = e
                    affinity[x * height + y][(x + i) * height + y + j] = 1 / (1 + max_edge)
    clusters, res = clustering.process_points(pts, affinity, k, 0.001)
    # clustering.visualize_result(res, 2, x)

    rgbs = [(255, 0, 0), (0, 255, 0), (255, 255, 0)]
    new_im = np.empty((width, height, 3), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            for i in range(k):
                if clusters[x * height + y][i] == 1: break
            r, g, b = rgbs[i]
            new_im[x][y][0] = (im[x][y] + r) / 2
            new_im[x][y][1] = (im[x][y] + g) / 2
            new_im[x][y][2] = (im[x][y] + b) / 2
    Image.fromarray(new_im).show()