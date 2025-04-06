import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super(UNET, self).__init__()
        self.upsampling = nn.ModuleList()
        self.downsampling = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Downsampling part of UNET
        for feature in features:
            self.downsampling.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Upsampling part of UNET
        for feature in reversed(features):
            self.upsampling.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.upsampling.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downsampling:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x) # Reduce the size of the image

        x = self.bottleneck(x) #The final step between the convolution layer and the dense layer

        #Reverse the skip connections list for use on the encoder side of UNET
        skip_connections = skip_connections[::-1]

        # Iterate through the transconvolution and double conv upsample layers with a step of 2
        for idx in range(0, len(self.upsampling), 2):
            x = self.upsampling[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_skip = torch.concat((skip_connection, x), dim=1) #TODO LEARN
            x = self.upsampling[idx+1](concat_skip)

        return self.final_conv(x)

