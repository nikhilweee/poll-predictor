import random
from utils import dictobject

def zeros(n):
    return [0] * int(n)

def arr_contains(arr, elt):
    if elt in arr:
        return True
    else:
        return False

def arr_unique(arr):
    return list(set(arr))

def maxmin(w):
    if not w:
        return []
    d = dictobject({})
    d.maxv = max(w)
    d.maxi = w.index(maxv)
    d.minv = min(w)
    d.mini = w.index(minv)
    d.dv = maxv - minv
    return d

def randperm(n):
    l = [x for x in range(n)]
    random.shuffle(l)
    return l
