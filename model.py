python


class AxialAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(AxialAttention, self).__init__()
        self.heads = heads
        self.scale = heads ** -0.5
        self.query = nn.Conv1d(in_channels, in_channels, 1, groups=heads)
        self.key = nn.Conv1d(in_channels, in_channels, 1, groups=heads)
        self.value = nn.Conv1d(in_channels, in_channels, 1, groups=heads)

    def forward(self, x):
        B, C, H, W, D = x.size()
        queries = self.query(x).view(B, self.heads, C // self.heads, H, W, D)
        keys = self.key(x).view(B, self.heads, C // self.heads, H, W, D)
        values = self.value(x).view(B, self.heads, C // self.heads, H, W, D)

        attn_scores = (queries @ keys.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = attn_probs @ values
        out = out.contiguous().view(B, C, H, W, D)
        return out

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_gn=False, axial_attention=False):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(out_channels // 2, out_channels) if use_gn else nn.BatchNorm3d(out_channels)
        self.norm2 = nn.GroupNorm(out_channels // 2, out_channels) if use_gn else nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.01)
        self.axial_attention = AxialAttention(out_channels) if axial_attention else None

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        if self.axial_attention:
            x = x + self.axial_attention(x)
        return x

......
