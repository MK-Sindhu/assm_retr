import os
import random
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class PartGraphDataset(torch.utils.data.Dataset):
    def __init__(self, parts_dir):
        self.parts_dir = parts_dir

        self.files = [
            f for f in os.listdir(parts_dir)
            if f.lower().endswith(".pt")
        ]

        # Map part_id → list of occurs (positives)
        self.groups = {}
        for f in self.files:
            part_id = f.split("_")[0]  # before first underscore
            self.groups.setdefault(part_id, []).append(f)

        self.part_ids = list(self.groups.keys())

        print(f"Loaded {len(self.files)} part graphs")
        print(f"{len(self.part_ids)} unique part IDs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Return triplet (anchor, pos, neg)."""

        # Anchor
        anchor_file = self.files[idx]
        anchor_pid = anchor_file.split("_")[0]
        anchor_graph = torch.load(os.path.join(self.parts_dir, anchor_file))

        # Positive (same part)
        pos_file = random.choice(self.groups[anchor_pid])
        pos_graph = torch.load(os.path.join(self.parts_dir, pos_file))

        # Negative (random different part)
        neg_pid = random.choice([p for p in self.part_ids if p != anchor_pid])
        neg_file = random.choice(self.groups[neg_pid])
        neg_graph = torch.load(os.path.join(self.parts_dir, neg_file))

        return anchor_graph, pos_graph, neg_graph
