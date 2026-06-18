import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DRUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)

        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        def match_size(a, b):
            if a.shape != b.shape:
                a = F.interpolate(a, size=b.shape[2:], mode='bilinear', align_corners=False)
            return a

        d4 = self.dec4(torch.cat([match_size(self.up4(b), e4), e4], dim=1))
        d3 = self.dec3(torch.cat([match_size(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([match_size(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([match_size(self.up1(d2), e1), e1], dim=1))
        return (x + self.head(d1)).clamp(0, 1)
