import torch
import torch.nn as nn

from torchvision import models
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
    )

class sin_seg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.b_to_3 = convrelu(in_channels, 3, 3, 1)

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512 + 256, 512, 3, 1)  # 64 for frequency domain
        self.conv_up2 = convrelu(128 + 512 + 128, 256, 3, 1)  # 64 for frequency domain
        self.conv_up1 = convrelu(64 + 256 + 64, 256, 3, 1)  # 64 for frequency domain
        self.conv_up0 = convrelu(64 + 256 + 64, 128, 3, 1)  # 64 for frequency domain

        self.freq_up_recon = convrelu(256, 1, 3, 1)
        self.align = convrelu(192, in_channels, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, out_channels, 1)
        self.conv_recon_last = convrelu(256+64, 1, 3, 1)

        self.enhance_up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.enhance_up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.enhance_up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.enhance_up3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.enhance_up4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.conv_core0 = convrelu(1024, 256, 3, 1)

        # self.mlp = MLP(512*8*8, 512)
        self.fc = nn.Linear(512*8*8, 512)
        # self.norm = nn.LayerNorm(512)
        self.norm = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, freq_feature):
        input = self.b_to_3(input)
        freq_feature = self.enhance_up2(freq_feature)
        freq_feature = self.align(freq_feature)

        x_original = self.conv_original_size0(input)  
        freq_original = self.conv_original_size0(freq_feature)
        x_original = self.conv_original_size1(x_original)  
        freq_original = self.conv_original_size1(freq_original)

        layer0 = self.layer0(input)  
        freq0 = self.layer0(freq_feature)  

        layer1 = self.layer1(layer0) 
        freq1 = self.layer1(freq0)
        layer2 = self.layer2(layer1)  
        freq2 = self.layer2(freq1)
        layer3 = self.layer3(layer2) 
        freq3 = self.layer3(freq2)
        layer4 = self.layer4(layer3)  
        freq4 = self.layer4(freq3)

        layer4 = self.layer4_1x1(layer4)  
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)  
        freq3 = self.layer3_1x1(freq3)

        x = torch.cat([x, layer3, freq3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        freq2 = self.layer2_1x1(freq2)
        x = torch.cat([x, layer2, freq2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        freq1 = self.layer1_1x1(freq1)
        x = torch.cat([x, layer1, freq1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)  
        freq0 = self.layer0_1x1(freq0)
        x = torch.cat([x, layer0, freq0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)  
        x = self.conv_original_size2(x)  

        out = self.conv_last(x)

        layer0_enhance = self.enhance_up0(freq0)  
        layer1_enhance = self.enhance_up1(freq1)  
        layer2_enhance = self.enhance_up2(freq2)  
        layer3_enhance = self.enhance_up3(freq3)  
        layer4_enhance = self.enhance_up4(freq4)  
        core_enhance = torch.cat([layer0_enhance, layer1_enhance,
                                  layer2_enhance, layer3_enhance, layer4_enhance], dim=1) 
        core_enhance = self.conv_core0(core_enhance)  
        freq_recon_out = torch.cat([core_enhance, freq_original], dim=1)  
        freq_recon_out = self.conv_recon_last(freq_recon_out)

        freq4 = freq4.view(freq4.size(0), -1)
        freq_feature = self.fc(freq4)
        freq_feature = self.norm(freq_feature)
        freq_feature = self.relu(freq_feature)
        freq_feature = freq_feature.view(freq_feature.size(0), -1, 1)

        layer4 = layer4.view(layer4.size(0), -1)
        spatial_feature = self.fc(layer4)
        spatial_feature = self.norm(spatial_feature)
        spatial_feature = self.relu(spatial_feature)
        spatial_feature = spatial_feature.view(spatial_feature.size(0), 1, -1)

        align_output = torch.bmm(freq_feature, spatial_feature)
        return out, freq_recon_out, align_output