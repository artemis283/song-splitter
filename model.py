# model will be a convUnet with an encoder and decoder 
# encoder will be layers of conv2d + batchnorm _ relu + downsample (maxpool or stride 2) 
# bottleneck is conv2d layers that can be swapped out for transformer eventually
# decoder will be upsample (convtranspose2d or billinear) + skip connectiosns, conv2d + batchnnorm + Relu to original resolution
# output is final conv2d layer to output mask or magnitude

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class EncoderBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(EncoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        # Add residual connection
        x = x + residual
        x = nn.ReLU(inplace=True)(x)
        
        return x
    

    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(DecoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        # Add residual connection
        x = x + residual
        x = nn.ReLU(inplace=True)(x)
        
        return x
        
           

class ConvUNet(nn.Module):
    def __init__(self, out_channels=1):
        super(ConvUNet, self).__init__()
        self.use_checkpoint = False

        # Increased number of channels
        self.encoders = nn.ModuleList([
            EncoderBlock(1, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
            EncoderBlock(256, 512),
            EncoderBlock(512, 1024),
        ])

        self.decoders = nn.ModuleList([
            DecoderBlock(1024, 512),
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
        ])

        # Enhanced bottleneck with more layers
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        ])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def use_checkpointing(self):
        self.use_checkpoint = True

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            if self.use_checkpoint and self.training:
                x = checkpoint(encoder, x)
            else:
                x = encoder(x)
            if encoder != self.encoders[-1]:
                skips.append(x)
                x = self.downsample(x)

        if self.use_checkpoint and self.training:
            x = checkpoint(self.bottleneck, x)
        else:
            x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            x = self.upsamplers[i](x)
            skip = skips[-(i+1)]
            x = torch.cat((x, skip), dim=1)
            if self.use_checkpoint and self.training:
                x = checkpoint(decoder, x)
            else:
                x = decoder(x)

        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    model = ConvUNet()
    print(model)

    x = torch.randn(1, 1, 256, 256)
    print(model(x).shape)
    
        
        
