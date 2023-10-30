import torch
from torchdiffeq import odeint_adjoint as odeint

class SimpleFunction(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = torch.nn.Parameter(a)
        x0 = torch.ones_like(a)
        t = torch.linspace(0, 1, 1000, dtype=a.dtype, device=a.device)
        xt = odeint(self, x0, t)
        # self.xt = xt  # NO_MEMLEAK_IF: this line is removed

    def forward(self, t, x):
        return -self.a * x

def test_fcn():
    a = torch.ones((300000,), dtype=torch.double, device=torch.device("cuda"))
    model = SimpleFunction(a)

for i in range(5):
    test_fcn()
    # torch.cuda.empty_cache()
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')