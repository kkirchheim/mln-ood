import torch
import torch.nn as nn
from torch import Tensor
import logging

log = logging.getLogger(__name__)


class MLN(nn.Module):
    """
    A simple markov logic network
    """

    def __init__(self, constraints, domain=None):
        super(MLN, self).__init__()
        self.constraints = constraints
        self.w = nn.Parameter(
            torch.randn(size=(len(self.constraints),), dtype=torch.double),
            requires_grad=True,
        )
        torch.nn.init.constant_(self.w.data, 0.0)
        self.domain = domain

        if not domain:
            raise ValueError

        self.state_space = None
        # log.info(f"State space size: {self.state_space.size()}, {self.state_space.dtype} {self.state_space.shape[1] * self.state_space.shape[0] * 8 / 1024 / 1024}MB")
        self.domain = domain

    def to(self, device):
        """
        Offload state space to device
        """
        super().to(device)
        # self.state_space = self.state_space.to(device)

    def calc_z(self, device="cpu"):
        """
        Calculate value of partition function.

        Only allocate state space on demand
        """
        if self.state_space is None:
            self.state_space = torch.cartesian_prod(
                *[torch.tensor(d, dtype=torch.uint8) for d in self.domain]
            )

        self.state_space = self.state_space.to(device)

        z = self.energy(self.state_space).sum()

        return z

    def energy(self, x):
        e = torch.zeros(size=(x.shape[0], 1), device=x.device, dtype=torch.double)
        for i, f in enumerate(self.constraints):
            fx = f(x).double()
            e += fx * self.w[i]

        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        max_e = torch.max(e)
        e_stable = e - max_e
        e_exp = e_stable.exp()
        e_final = e_exp.log() + max_e
        return e_final.exp().squeeze()

    # def forward(self, x):
    #     return self.prob(x)
    #
    def prob(self, x) -> Tensor:
        energy = self.energy(x).double()
        z = self.calc_z(x.device).double()
        return energy / z
