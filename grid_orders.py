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
        related_size = width
    elif t == 1:
        a = _j * M + _i
        related_size = height
    elif t == 2:
        a = _i * M + inverse_j
        related_size = width
    elif t == 3:
        a = inverse_j * M + _i
        related_size = height
    elif t == 4:
        a = inverse_i * M + inverse_j
        related_size = width
    elif t == 5:
        a = inverse_j * M + inverse_i
        related_size = height
    elif t == 6:
        a = inverse_i * M + _j
        related_size = width
    elif t == 7:
        a = _j * M + inverse_i
        related_size = height
    elif t == 8:
        a = (_i + _j) * M + inverse_i
        related_size = min(width, height)
    elif t == 9:
        a = (_i + _j) * M + _i
        related_size = min(width, height)
    elif t == 10:
        a = (_i + inverse_j) * M + inverse_i
        related_size = min(width, height)
    elif t == 11:
        a = (_i + inverse_j) * M + _i
        related_size = min(width, height)
    elif t == 12:
        a = (inverse_i + inverse_j) * M + _i
        related_size = min(width, height)
    elif t == 13:
        a = (inverse_i + inverse_j) * M + inverse_i
        related_size = min(width, height)
    elif t == 14:
        a = (inverse_i + _j) * M + _i
        related_size = min(width, height)
    elif t == 15:
        a = (inverse_i + _j) * M + inverse_i
        related_size = min(width, height)
    else:
        print("Wrong value for t in grid_orders.get_random_order!")
        
    ind = np.argsort(a.reshape([-1]))
    pi = np.zeros([height*width], dtype=np.int)
    pi[ind] = np.arange(height*width, dtype=np.int)
    #print(t,':')
    #print(pi.reshape([height, width]))
    return [pi, related_size]
        