python

# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, use_gn=False, axial_attention_levels=[]):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = UNetBlock(in_channels, 32, use_gn)
        self.enc2 = UNetBlock(32, 64, use_gn)
        self.enc3 = UNetBlock(64, 128, use_gn)
        self.enc4 = UNetBlock(128, 256, use_gn, axial_attention=(1 in axial_attention_levels))
        self.enc5 = UNetBlock(256, 512, use_gn, axial_attention=(2 in axial_attention_levels))

        # Decoder
        self.dec1 = UNetBlock(512 + 256, 256, use_gn, axial_attention=(3 in axial_attention_levels))
        self.dec2 = UNetBlock(256 + 128, 128, use_gn, axial_attention=(4 in axial_attention_levels))
        self.dec3 = UNetBlock(128 + 64, 64, use_gn)
        self.dec4 = UNetBlock(64 + 32, 32, use_gn)

        # Final Convolution
        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool3d(x1, 2))
        x3 = self.enc3(F.max_pool3d(x2, 2))
        x4 = self.enc4(F.max_pool3d(x3, 2))
        x5 = self.enc5(F.max_pool3d(x4, 2))

        # Decoder
        x = F.interpolate(x5, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.dec1(torch.cat([x, x4], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.dec2(torch.cat([x, x3], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.dec3(torch.cat([x, x2], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.dec4(torch.cat([x,
