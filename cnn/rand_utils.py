import random
import math

def gauss_random(return_v,v_val):
  if return_v :
    return_v = False
    return v_val 
  u = 2*random.random()-1
  v = 2*random.random()-1
  r = u*u + v*v
  if r == 0 or r > 1 :
    return gauss_random(return_v,v_val)
  c = math.sqrt(-2*math.log(r)/r)
  v_val = v*c # cache this
  return_v = True
  return u*c

def randf(a, b): 
  return random.random()*(b-a)+a

def randi(a, b):
  return math.floor(random.random()*(b-a)+a)

def randn(mu, std):
  return mu + gauss_random(False, 0.0)*std 