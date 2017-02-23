import numpy as np
from scipy import ndimage
from scipy import misc
from PIL import Image
import math
import clustering
import data_reader

city_block_radius = 5

def test_edge():
    im = ndimage.imread("../data/plane.jpg", True)

    sx = ndimage.sobel(im, axis=0, mode='constant')
    sy = ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)

    # Image.fromarray(sob).show()

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
    x, res = clustering.process_points(pts, affinity, 2, 0.001)
    clustering.visualize_result(res, 2, x)
    # out = np.zeros((width, height))
    # max_sum = 0
    # for x in range(width):
    #     for y in range(height):
    #         for r in range(width * height):
    #             out[x][y] += affinity[x * height + y][r]
    #         if out[x][y] > max_sum: max_sum = out[x][y]
    # out = out / max_sum * 255;
    # Image.fromarray(out).show()

    # print(affinity)
