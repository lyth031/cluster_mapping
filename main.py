import renet
import torch
import time

rwl = 3.9
rbl = 3.9
a = renet.Array()
w = torch.zeros([32, 32])
vin = 0.15*torch.ones([1, 32])
rram = a.setValue(w)
print(vin)
ideal = a.idealOuput(vin)
print(ideal*1e6)
"""
real = a.realOutput(vin)
print(real*1e6)
"""
start = time.time()
real_comparison = a.realOutputComparison(vin)
print(real_comparison*1e6)
end = time.time()
print (end-start)
"""
print(real - real_comparison)
"""