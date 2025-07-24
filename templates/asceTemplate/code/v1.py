import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialEncoder(nn.Module):
    def __init__(self, input_length, enc_dim=32):
        super(SpatialEncoder, self).__init__()
        # 1D convolutions for spatial feature extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
        self.fc = nn.Linear(16 * input_length, enc_dim)
    
    def forward(self, x):
        # x shape: (batch, 1, input_length)
        feat = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(feat))
        # Flatten
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        return F.relu(feat)

class PINN_CNN_LSTM(nn.Module):
    def __init__(self, input_length, enc_dim=32, hidden_dim=64):
        super(PINN_CNN_LSTM, self).__init__()
        self.encoder = SpatialEncoder(input_length, enc_dim)
        self.lstm = nn.LSTM(input_size=enc_dim, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, input_length)
    
    def forward(self, initial_state, n_steps):
        # initial_state: (batch, 1, input_length)
        batch_size = initial_state.size(0)
        
        # Encode initial spatial state
        h0_input = self.encoder(initial_state)  # (batch, enc_dim)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size)
        # Simple init of hidden using encoder output
        h0[0] = h0_input
        
        outputs = []
        current_state = initial_state
        
        for t in range(n_steps):
            feat = self.encoder(current_state)    # (batch, enc_dim)
            lstm_in = feat.unsqueeze(1)           # (batch, 1, enc_dim)
            out_seq, (h0, c0) = self.lstm(lstm_in, (h0, c0))
            pred_field = self.decoder(out_seq[:, -1, :])  # (batch, input_length)
            pred_field = pred_field.unsqueeze(1)          # (batch, 1, input_length)
            outputs.append(pred_field)
            current_state = pred_field
        
        return outputs
