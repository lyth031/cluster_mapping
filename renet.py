import torch
import math

class Array(object):
    """A nwl x nbl resistive cross-point array."""

    # Constructor
    def __init__(self,nwl = 32, nbl = 32, rlrs = 50e3, rhrs = 500e3, rwl = 3.9, rbl = 3.9):
        self.nwl = nwl
        self.nbl = nbl
        self.rlrs = rlrs
        self.rhrs = rhrs
        self.rwl = rwl
        self.rbl = rbl
        self.vnode = torch.zeros([2,self.nwl,self.nbl], dtype=torch.float)
        self.rram = torch.zeros([self.nwl,self.nbl], dtype=torch.float)
        
    def setValue(self, w):
        self.rram = 1 /( 1 / self.rhrs + (1 / self.rlrs - 1 / self.rhrs) * w)
        return self.rram

    # Ideal output
    def idealOuput(self,vin):   
        return torch.mm(vin, 1 / self.rram)

    # Real output
    def realOutput(self,vin):
        tol = 1e-7
        max_step = 1e5
        error = 1
        count = 0
        w=4/(2+math.sqrt(4-(math.cos(math.pi/self.nwl)+math.cos(math.pi/self.nbl))**2))
        left = 2 / self.rwl * torch.ones(self.nwl, self.nbl-1)
        right = 1 / self.rwl * torch.ones(self.nwl, 1)       
        relax_rwl = torch.cat((left, right),1) +  1 / self.rram
        top = 1 / self.rbl * torch.ones(1, self.nbl)
        bottom = 2 / self.rbl * torch.ones(self.nbl-1, self.nbl)
        relax_rbl = torch.cat((top, bottom), 0) + 1 / self.rram
        self.vnode[0] = torch.mm(vin.t(), torch.ones([1,self.nbl]))

        while error > tol and count <= max_step:
            error = 0
            left = (vin.t() + self.vnode[0, :, 1:2]) / self.rwl
            mid = (self.vnode[0, :, 0:self.nbl-2] + self.vnode[0, :, 2:self.nbl]) / self.rwl
            right = self.vnode[0, :, self.nbl-2:self.nbl-1] / self.rwl
            relax = torch.cat((left, mid ,right),1) + torch.div(self.vnode[1], self.rram)
            relax = w * (torch.div(relax, relax_rwl) - self.vnode[0])
            self.vnode[0] += relax
            m = relax.abs().max()
            if error <= m:
                error = m
            
            top = self.vnode[1, 1:2, :] / self.rbl
            mid = (self.vnode[1, 0:self.nwl-2, :] + self.vnode[1, 2:self.nwl, :]) / self.rbl 
            bottom = self.vnode[1, self.nwl-2:self.nwl-1, :] / self.rbl
            relax = torch.cat((top, mid ,bottom),0) + torch.div(self.vnode[0], self.rram)
            relax = w * (torch.div(relax, relax_rbl) - self.vnode[1])
            self.vnode[1] += relax
            m = relax.abs().max()
            if error <= m:
                error = m
            
            count += 1
        print(self.vnode)
        vrram = self.vnode[0] - self.vnode[1]
        point_out = torch.div(vrram, self.rram)
        out = torch.sum(point_out,0)
        return out.unsqueeze(0)

    # Real output comparison
    def realOutputComparison(self,vin):
        tol = 2e-7
        max_step = 1e5
        error = 1
        count = 0
        w=4/(2+math.sqrt(4-(math.cos(math.pi/self.nwl)+math.cos(math.pi/self.nbl))**2))
        self.vnode[0] = torch.mm(vin.t(), torch.ones([1,self.nbl]))
        print(self.vnode)

        while error > tol and count <= max_step:
            error = 0
            for i in range(self.nwl):
                for j in range(self.nbl):
                    if j == 0:
                        relax = w * ((self.vnode[0,i,j+1] / self.rwl + vin[0, i] / self.rwl + self.vnode[1, i, j] / self.rram[i, j]) / (2 / self.rwl + 1 / self.rram[i, j]) - self.vnode[0, i, j])
                    elif j == self.nbl-1:
                        relax = w * ((self.vnode[0,i,j-1] / self.rwl + self.vnode[1, i, j] / self.rram[i, j]) / (1 / self.rwl + 1 / self.rram[i, j]) - self.vnode[0, i, j])
                    else:
                        relax = w * ((self.vnode[0,i,j+1] / self.rwl + self.vnode[0,i,j-1] / self.rwl + self.vnode[1, i, j] / self.rram[i, j]) / (2 / self.rwl + 1 / self.rram[i, j]) - self.vnode[0, i, j])
                    
                    self.vnode[0, i, j] += relax
                    m = abs(relax)
                    if error <= m:
                        error = m
                    
                    if i == 0:
                        relax = w * ((self.vnode[1,i+1,j] / self.rbl + self.vnode[0, i, j] / self.rram[i, j]) / (1 / self.rbl + 1 / self.rram[i, j]) - self.vnode[1, i, j])
                    elif i == self.nwl-1:
                        relax = w * ((self.vnode[1,i-1,j] / self.rbl + self.vnode[0, i, j] / self.rram[i, j]) / (2 / self.rbl + 1 / self.rram[i, j]) - self.vnode[1, i, j])
                    else:
                        relax = w * ((self.vnode[1,i+1,j] / self.rbl + self.vnode[1,i-1,j] / self.rbl + self.vnode[0, i, j] / self.rram[i, j]) / (2 / self.rbl + 1 / self.rram[i, j]) - self.vnode[1, i, j])

                    self.vnode[1, i, j] += relax
                    m = abs(relax)
                    if error <= m:
                        error = m                
            
            count += 1
        
        vrram = self.vnode[0] - self.vnode[1]
        point_out = torch.div(vrram, self.rram)
        out = torch.sum(point_out,0)
        return out.unsqueeze(0)

    # Test
    def print(self):
        print(self.vnode)