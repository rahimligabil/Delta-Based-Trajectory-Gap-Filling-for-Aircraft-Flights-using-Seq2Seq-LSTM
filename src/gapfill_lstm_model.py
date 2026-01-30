import torch
import torch.nn as nn

# -----------------------------
# Encoder: (B,6,8) -> (h,c)
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        # x: (B,6,7)
        _, (h, c) = self.lstm(x)
        # h,c: (L,B,H)
        return h, c


# -----------------------------
# Decoder: step-by-step generate
# input: (B,1,3) + (h,c) -> (B,1,3)
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, output_size=3, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_step, h, c):
        # x_step: (B,1,3)
        out, (h, c) = self.lstm(x_step, (h, c))
        # out: (B,1,H)
        y_step = self.fc(out)  # (B,1,3)
        return y_step, h, c


# -----------------------------
# Two-Encoder Seq2Seq Model
# x_in: (B,12,8) -> y_pred: (B,8,3)
# -----------------------------
class TwoEncoderSeq2SeqGapFill(nn.Module):
    def __init__(self, input_size=8, output_size=3, hidden_size=128, num_layers=2, dropout=0.1,
                 past_len=6, future_len=6, gap_len=8):
        super().__init__()

        self.past_len = past_len
        self.future_len = future_len
        self.gap_len = gap_len

        self.encoder_past = Encoder(input_size, hidden_size, num_layers, dropout)
        self.encoder_future = Encoder(input_size, hidden_size, num_layers, dropout)

        # Fuse past/future states per layer: (2H) -> (H)
        self.fuse_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fuse_c = nn.Linear(hidden_size * 2, hidden_size)

        self.decoder = Decoder(output_size, hidden_size, num_layers, dropout)

    def forward(self, x_in, y_true=None, teacher_forcing=0.7):
        """
        x_in:   (B,12,8)  -> [past(6), future(6)]
        y_true: (B,8,3)   -> provided during training for teacher forcing
        """
        B = x_in.size(0)
        device = x_in.device

        # 1) split
        past   = x_in[:, :self.past_len, :]                 # (B,6,8)
        future = x_in[:, self.past_len:self.past_len+self.future_len, :]  # (B,6,8)

        # 2) encode separately
        h_p, c_p = self.encoder_past(past)    # (L,B,H)
        h_f, c_f = self.encoder_future(future)# (L,B,H)

        # 3) fuse states layer-by-layer
        h_cat = torch.cat([h_p, h_f], dim=2)  # (L,B,2H)
        c_cat = torch.cat([c_p, c_f], dim=2)  # (L,B,2H)

        h = self.fuse_h(h_cat)  # (L,B,H)
        c = self.fuse_c(c_cat)  # (L,B,H)

        # 4) decode step-by-step
        prev = torch.zeros((B, 1, 3), device=device)  # start token
        outputs = []

        for t in range(self.gap_len):
            y_step, h, c = self.decoder(prev, h, c)  # (B,1,3)
            outputs.append(y_step)

            # teacher forcing
            if (y_true is not None) and (torch.rand(1).item() < teacher_forcing):
                prev = y_true[:, t:t+1, :]   # (B,1,3)
            else:
                prev = y_step.detach()

        return torch.cat(outputs, dim=1)  # (B,8,3)
