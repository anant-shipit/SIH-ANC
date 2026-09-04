import torch
import torch.nn as nn

class StubGTCRN(nn.Module):
    """A minimal GRU-based PyTorch module pretending to be GTCRN.
    
    This exercises the streaming export IO discovery (identifying the hidden 
    states) and cross-frame state propagation in integration tests, without 
    needing the actual 48K-param GTCRN.
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        # The input is (batch=1, freq=257, time=1, complex=2)
        # We'll flatten it to a feature vector of size 257 * 2 = 514
        self.gru = nn.GRU(514, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 514)
        
    def forward(self, spec_frame, *states):
        # spec_frame shape: (batch=1, freq=257, time=1, complex=2)
        batch, freq, time, comp = spec_frame.shape
        # Move time to dim 1, then flatten freq and comp
        x = spec_frame.permute(0, 2, 1, 3).reshape(batch, time, freq * comp)
        
        if len(states) > 0:
            h0 = states[0]
            out, hn = self.gru(x, h0)
        else:
            out, hn = self.gru(x)
            
        out = self.fc(out)
        # Reshape back to (batch, time, freq, comp) and permute to (batch, freq, time, comp)
        out = out.reshape(batch, time, freq, comp).permute(0, 2, 1, 3)
        
        if len(states) > 0:
            return out, hn
        return out
        
def create_stub_model():
    torch.manual_seed(42)
    return StubGTCRN()
