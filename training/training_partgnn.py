
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


from data_loading.parts_dataset import PartGraphDataset
from models.part_gnn import PartGNN

# Safety for torch.load (PyTorch 2.6+)
import torch.serialization
import torch_geometric

torch.serialization.add_safe_globals([
    torch_geometric.data.Data,
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage,
])

# ---------------- CONFIG ----------------
PART_DIR = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/MK_S/graphs/parts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 1      # Test run, increase later
EMBED_DIM = 256
LR = 1e-3
CKPT = "partgnn_checkpoint.pt"
print("Training on:", DEVICE)

# --------------- Dataset ----------------
dataset = PartGraphDataset(PART_DIR)

# Build model
sample = dataset[0][0]
model = PartGNN(in_channels=sample.x.size(1), embed_dim=EMBED_DIM).to(DEVICE)

if os.path.exists(CKPT):
    print("Loading checkpoint:", CKPT)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

# Custom batching for triplet data
from torch_geometric.data import Batch

def triplet_collate(batch):
    anchors = [item[0] for item in batch]
    positives = [item[1] for item in batch]
    negatives = [item[2] for item in batch]

    return (
        Batch.from_data_list(anchors),
        Batch.from_data_list(positives),
        Batch.from_data_list(negatives)
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=triplet_collate
)

# ---------------- Training ----------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    for batch in pbar:
        anchors, positives, negatives = batch

        anchors = anchors.to(DEVICE)
        positives = positives.to(DEVICE)
        negatives = negatives.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        emb_a = model(anchors.x, anchors.edge_index, anchors.batch)
        emb_p = model(positives.x, positives.edge_index, positives.batch)
        emb_n = model(negatives.x, negatives.edge_index, negatives.batch)

        loss = triplet_loss(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

    torch.save(model.state_dict(), CKPT)
    print(f"Checkpoint saved → {CKPT}")

print("\n🎯 Training complete!")



