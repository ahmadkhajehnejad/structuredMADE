import numpy as np

def get_random_order(width, height):
    _i = np.tile(np.arange(height).reshape([-1,1]), [1,width])
    _j = np.tile(np.arange(width).reshape([1,-1]), [height,1])
    M = width * height
    inverse_i = (height - 1 - _i)
    inverse_j = (width - 1 - _j)
    t = np.random.randint(16)
    if t == 0:
        a = _i * M + _j
    elif t == 1:
        a = _j * M + _i
    elif t == 2:
        a = _i * M + inverse_j
    elif t == 3:
        a = inverse_j * M + _i
    elif t == 4:
        a = inverse_i * M + inverse_j
    elif t == 5:
        a = inverse_j * M + inverse_i
    elif t == 6:
        a = inverse_i * M + _j
    elif t == 7:
        a = _j * M + inverse_i
    elif t == 8:
        a = (_i + _j) * M + inverse_i
    elif t == 9:
        a = (_i + _j) * M + _i
    elif t == 10:
        a = (_i + inverse_j) * M + inverse_i
    elif t == 11:
        a = (_i + inverse_j) * M + _i
    elif t == 12:
        a = (inverse_i + inverse_j) * M + _i
    elif t == 13:
        a = (inverse_i + inverse_j) * M + inverse_i
    elif t == 14:
        a = (inverse_i + _j) * M + _i
    elif t == 15:
        a = (inverse_i + _j) * M + inverse_i
    else:
        print("Wrong value for t in grid_orders.get_random_order!")



    ind = np.argsort(a.reshape([-1]))
    pi = np.zeros([height*width], dtype=np.int)
    pi[ind] = np.arange(height*width, dtype=np.int)
    return pi


def get_random_order_from_center(width, height):
    a = get_random_order(width, height)

    i_c = np.random.randint(int(height//3), height - int(height//3))
    j_c = np.random.randint(int(width//3), width - int(width//3))
    k = a.reshape([height, width])[i_c, j_c]

    ind_1 = a < k
    ind_2 = a >= k
    pi_1 = a[ind_1]
    pi_2 = a[ind_2]
    if (np.random.rand() > 0.5):
        pi_1 = (k - pi_1) * 2
        pi_2 = (pi_2 - k) * 2 - 1
        pi_2[pi_2 == -1] = 0
    else:
        pi_2 = (pi_2 - k) * 2
        pi_1 = (k - pi_1) * 2 - 1
    a[ind_1] = pi_1
    a[ind_2] = pi_2

    ind = np.argsort(a.reshape([-1]))
    pi = np.zeros([height * width], dtype=np.int)
    pi[ind] = np.arange(height*width, dtype=np.int)
    return pi


def _get_partially_permuted(pi, num_parts):
    n = len(pi)
    sz = int(n // num_parts)
    part_size = np.ones(num_parts, dtype=int) * sz
    part_size[:n - (sz * num_parts)] += 1
    points = np.concatenate([[0], np.cumsum(part_size)])
    for k in range(num_parts):
        ind = (points[k] <= pi) & (pi < points[k+1])
        tmp = np.arange(points[k], points[k+1])
        np.random.shuffle(tmp)
        pi[ind] = tmp
    return pi

def get_partially_random_order(width, height, num_parts, initial_fixed_order=False):
    if initial_fixed_order:
        pi = np.arange(width*height)
    else:
        pi = get_random_order(width, height)
    return _get_partially_permuted(pi, num_parts)
