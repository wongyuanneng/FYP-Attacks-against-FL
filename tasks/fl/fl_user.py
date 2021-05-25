from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader
from defences.pdgan import PDGAN


@dataclass
class FLUser:
    user_id: int = 0
    compromised: bool = False
    train_loader: DataLoader = None


@dataclass
class FLServer:
    user_id: int = 0
    compromised: bool = False
    train_loader: DataLoader = None
    defence_utility: PDGAN = None