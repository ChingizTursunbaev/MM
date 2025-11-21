import torch
import torch.nn as nn
import math

# Import Mamba-2 or Mamba-1
try:
    from mamba_ssm import Mamba2
    MAMBA_CLS = Mamba2
    print("Using Mamba-2 Backbone.")
except ImportError:
    try:
        from mamba_ssm import Mamba
        MAMBA_CLS = Mamba
        print("Using Mamba-1 Backbone.")
    except ImportError:
        MAMBA_CLS = None
        print("WARNING: Mamba not found. Install 'mamba-ssm'.")

class SpatialQMamba(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 num_joints: int = 133,
                 token_embed_dim: int = 8, 
                 model_dim: int = 512, 
                 compress_factor: int = 4,
                 num_layers: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        # --- 1. Geometric Embeddings (Option B) ---
        # Table 1: Token Semantics (1-9)
        self.token_emb = nn.Embedding(10, token_embed_dim, padding_idx=0)
        self._init_geometric_weights()
        
        # Table 2: Joint Identity (133 joints)
        # We add this to the token embedding so the model knows "Who" moved.
        self.joint_id_emb = nn.Parameter(torch.randn(1, 1, num_joints, token_embed_dim) * 0.02)

        # Project combined features to Model Dimension
        # Input: 133 * token_dim -> Output: model_dim
        self.spatial_proj = nn.Linear(num_joints * token_embed_dim, model_dim)
        self.spatial_act = nn.GELU()

        # --- 2. Temporal Compressor Components ---
        # Reduces 15,000 frames -> 3,750 frames
        # We separate these because Conv1d and LayerNorm need different shapes
        self.conv_compress = nn.Conv1d(model_dim, model_dim, kernel_size=compress_factor, stride=compress_factor)
        self.norm_compress = nn.LayerNorm(model_dim)
        self.act_compress = nn.GELU()
        
        self.compress_factor = compress_factor

        # --- 3. Bi-Directional Backbone ---
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(BiDirectionalBlock(model_dim))
            
        self.norm = nn.LayerNorm(model_dim)
        
        # --- 4. CTC Head ---
        self.classifier = nn.Linear(model_dim, vocab_size)
        
        # --- 5. Dropout ---
        self.dropout = nn.Dropout(dropout)
    def _init_geometric_weights(self):
        """Initialize Token 5 as (+1, +1), Token 9 as (-1, -1), etc."""
        with torch.no_grad():
            self.token_emb.weight.data.normal_(0, 0.02)
            # (dx, dy) map for tokens 1-9
            directions = {
                1:(0,0), 2:(0,1), 3:(0,-1), 4:(1,0), 
                5:(1,1), 6:(1,-1), 7:(-1,0), 8:(-1,1), 9:(-1,-1)
            }
            for t, (dx, dy) in directions.items():
                # Use dim 0 for X, dim 1 for Y
                self.token_emb.weight.data[t, 0] = dx * 0.5
                self.token_emb.weight.data[t, 1] = dy * 0.5

    def forward(self, x, lengths):
        # x: (B, T_raw, 133)
        B, T, J = x.shape
        
        # A. Embed
        # (B, T, 133) -> (B, T, 133, D_emb)
        x = self.token_emb(x)
        
        # B. Add Joint IDs (Broadcasting over Time)
        # (B, T, 133, D_emb) + (1, 1, 133, D_emb)
        x = x + self.joint_id_emb
        
        # C. Spatial Mix
        # Flatten joints: (B, T, 133*D_emb)
        x = x.view(B, T, -1)
        x = self.spatial_act(self.spatial_proj(x)) # (B, T, D_model)
        x = self.dropout(x)
        
        # D. Temporal Compression (FIXED)
        # 1. Transpose for Conv1d: (B, Dim, T)
        x = x.transpose(1, 2) 
        x = self.conv_compress(x)
        
        # 2. Transpose BACK for LayerNorm/Mamba: (B, T_new, Dim)
        x = x.transpose(1, 2) 
        
        # 3. Apply Norm and Act
        x = self.act_compress(self.norm_compress(x))
        
        # Update lengths (Integer division)
        new_lengths = (lengths / self.compress_factor).long()
        new_lengths = torch.clamp(new_lengths, min=1) # Safety

        # E. Backbone
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        x = self.dropout(x)
        
        # F. Classifier
        logits = self.classifier(x) # (B, T_new, Vocab)
        
        # CTC expects LogSoftmax
        return nn.functional.log_softmax(logits, dim=-1), new_lengths

class BiDirectionalBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fwd = MAMBA_CLS(d_model=d_model, d_state=64, d_conv=4, expand=2)
        self.bwd = MAMBA_CLS(d_model=d_model, d_state=64, d_conv=4, expand=2)
        # Linear to fuse forward/backward
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x):
        # Forward Pass
        out_fwd = self.fwd(x)
        
        # Backward Pass (Flip, Process, Flip Back)
        out_bwd = self.bwd(torch.flip(x, [1]))
        out_bwd = torch.flip(out_bwd, [1])
        
        # Concatenate and Fuse
        combined = torch.cat([out_fwd, out_bwd], dim=-1)
        return x + self.fusion(combined) # Skip connection