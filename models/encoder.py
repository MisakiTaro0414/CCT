from models.backbones.resnet_backbone import ResNetBackbone
from utils.helpers import initialize_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

resnet50 = {
    "path": "models/backbones/pretrained/3x3resnet50-imagenet.pth",
}

class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class Encoder(nn.Module):
    def __init__(self, pretrained):
        super(Encoder, self).__init__()

        if pretrained and not os.path.isfile(resnet50["path"]):
            print("Downloading pretrained resnet (source : https://github.com/donnyyou/torchcv)")
            os.system('sh models/backbones/get_pretrained_model.sh')

        model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained=pretrained)
        self.base = nn.Sequential(
            nn.Sequential(model.prefix, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )

        # model = vgg19(pre_trained= pretrained, require_grad = True)
        # self.base = nn.Sequential(*model.seq_list[0:27]) # VGG base (512)
        # Please change the first argument of _PSPModule from 2048 to 512 to conduct this experiment
        # Please set num_out_ch to be 512 in model.py



        # model = timm.create_model("convnext_base_in22k", pretrained= True)
        # self.base = nn.Sequential(model.stem, model.stages, model.norm_pre)   # convnext_base_in22k (1024)
        # Please change the first argument of _PSPModule from 2048 to 1024 to conduct this experiment
        # Please set num_out_ch to be 1024 in model.py (Line 58-62)
        # Please set upscale to be 32 in model.py (Line 58-62)
        # Please change all 8s to 32 in up_sizes in trainer.py (Line 121) 



        #model = timm.create_model("poolformer_m36", pretrained= True)
        #self.base = nn.Sequential(model.patch_embed, model.network, model.norm) # poolformet m-36 (768)
        # Please change the first argument of _PSPModule from 2048 to 768 to conduct this experiment
        # Please set num_out_ch to be 768 in model.py (Line 58-62)
        # Please set upscale to be 32 in model.py (Line 58-62)
        # Please change all 8s to 32 in up_sizes in trainer.py (Line 121) 

        self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6])

    def forward(self, x):
        x = self.base(x)
        x = self.psp(x)
        return x

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()
