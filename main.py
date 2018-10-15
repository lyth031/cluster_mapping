import renet
import torch
import time

dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda: 0")
else:
    device = torch.device("cpu") 
a = renet.Array(device = device)
w = torch.ones([32, 32], device = device, dtype=torch.float)
start = time.time()
for i in range(1):
    vin = 0.15*torch.ones([1, 32], device = device, dtype=torch.float)
    rram = a.setValue(w)
    ideal = a.idealOuput(vin)
    real = a.realOutput(vin)
    print(real*1e6)
    # real_comparison = a.realOutputComparison(vin)
    # print(real - real_comparison)
    # print(real_comparison*1e6)

end = time.time()
print (end-start)
"""
print(real - real_comparison)
"""