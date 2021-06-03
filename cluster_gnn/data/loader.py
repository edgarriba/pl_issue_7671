from unittest.mock import Mock
from dataclasses import dataclass

import torch


@dataclass
class GraphData:
    x = torch.rand(1)  # shape, dtype ?
    edge_attr = torch.rand(1)  # shape, dtype ?
    edge_index = torch.rand(1)  # shape, dtype ?


class GraphDataModule(Mock):
    def __init__(self, data_root: str, num_workers: int) -> None:
        super().__init__(return_value=GraphData())
