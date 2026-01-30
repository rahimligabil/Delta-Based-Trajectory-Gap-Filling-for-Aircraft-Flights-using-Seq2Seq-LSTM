import torch
import torch.nn as nn
from gapfill_lstm_model import TwoEncoderSeq2SeqGapFill
from gapfill_dataloaders import create_dataloaders



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader = create_dataloaders(
    "data/X_gap_v3.npy",
    "data/Y_gap_v3.npy",
    batch_size=256
)

# model definition
model = TwoEncoderSeq2SeqGapFill(
    input_size=8,
    output_size=3,
    hidden_size=128,
    num_layers = 2,
    dropout = 0.1,
    past_len= 6,
    future_len=6,
    gap_len=8
).to(device)

# loss function 
criterion = torch.nn.MSELoss()

# optimizer AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)     # (B,12,8)
        y = y.to(device)     # (B,8,3)

        optimizer.zero_grad()

        y_pred = model(x, y_true=y, teacher_forcing=0.7)   # training mode with teacher forcing

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x, y_true=None, teacher_forcing=0.0)  # inference mode (no teacher forcing)

            loss = criterion(y_pred, y)
            total_loss += loss.item()

    return total_loss / len(loader)





num_epochs = 30

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss   = validate(model, val_loader, criterion)

    print(f"Epoch {epoch+1:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # save model checkpoint
    torch.save(model.state_dict(), "gapfill_model.pt")
