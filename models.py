import re
from torch import nn
from torchvision import models


class Model(nn.Module):
    def __init__(self, name="mobilenet-v3-small"):
        super(Model, self).__init__()

        if re.match(r"^alexnet$", name, re.IGNORECASE):
            self.model = models.alexnet(pretrained=True, progress=True)
        
        elif re.match(r"^vgg11$", name, re.IGNORECASE):
            self.model = models.vgg11(pretrained=True, progress=True)
        elif re.match(r"^vgg13$", name, re.IGNORECASE):
            self.model = models.vgg13(pretrained=True, progress=True)
        elif re.match(r"^vgg16$", name, re.IGNORECASE):
            self.model = models.vgg16(pretrained=True, progress=True)
        elif re.match(r"^vgg19$", name, re.IGNORECASE):
            self.model = models.vgg19(pretrained=True, progress=True)
        
        elif re.match(r"^vgg11_bn$", name, re.IGNORECASE):
            self.model = models.vgg11_bn(pretrained=True, progress=True)
        elif re.match(r"^vgg13_bn$", name, re.IGNORECASE):
            self.model = models.vgg13_bn(pretrained=True, progress=True)
        elif re.match(r"^vgg16_bn$", name, re.IGNORECASE):
            self.model = models.vgg16_bn(pretrained=True, progress=True)
        elif re.match(r"^vgg19_bn$", name, re.IGNORECASE):
            self.model = models.vgg19_bn(pretrained=True, progress=True)
        

        elif re.match(r"^resnet18$", name, re.IGNORECASE):
            self.model = models.resnet18(pretrained=True, progress=True)
        elif re.match(r"^resnet34$", name, re.IGNORECASE):
            self.model = models.resnet34(pretrained=True, progress=True)
        elif re.match(r"^resnet50$", name, re.IGNORECASE):
            self.model = models.resnet50(pretrained=True, progress=True)
        elif re.match(r"^resnet101$", name, re.IGNORECASE):
            self.model = models.resnet101(pretrained=True, progress=True)
        elif re.match(r"^resnet152$", name, re.IGNORECASE):
            self.model = models.resnet152(pretrained=True, progress=True)
        elif re.match(r"^wresnet50$", name, re.IGNORECASE):
            self.model = models.wide_resnet50_2(pretrained=True, progress=True)
        elif re.match(r"^wresnet101$", name, re.IGNORECASE):
            self.model = models.wide_resnet101_2(pretrained=True, progress=True)
        elif re.match(r"^resnext50$", name, re.IGNORECASE):
            self.model = models.resnext50_32x4d(pretrained=True, progress=True)
        elif re.match(r"^resnext101$", name, re.IGNORECASE):
            self.model = models.resnext101_32x8d(pretrained=True, progress=True)
        

        elif re.match(r"^densenet121$", name, re.IGNORECASE):
            self.model = models.densenet121(pretrained=True, progress=True)
        elif re.match(r"^densenet161$", name, re.IGNORECASE):
            self.model = models.densenet161(pretrained=True, progress=True)
        elif re.match(r"^densenet169$", name, re.IGNORECASE):
            self.model = models.densenet169(pretrained=True, progress=True)
        elif re.match(r"^densenet201$", name, re.IGNORECASE):
            self.model = models.densenet201(pretrained=True, progress=True)
        
        
        elif re.match(r"^mobilenet_v2$", name, re.IGNORECASE):
            self.model = models.mobilenet_v2(pretrained=True, progress=True)
        elif re.match(r"^mobilenet_v3_large$", name, re.IGNORECASE):
            self.model = models.mobilenet_v3_large(pretrained=True, progress=True)
        elif re.match(r"^mobilenet_v3_small$", name, re.IGNORECASE):
            self.model = models.mobilenet_v3_small(pretrained=True, progress=True)
    
        # elif re.match(r"^f_resnet50$", name, re.IGNORECASE):
        #     self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
        # elif re.match(r"^f_mobilenet$", name, re.IGNORECASE):
        #     self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, progress=True)
        # elif re.match(r"^f_mobilenet_320$", name, re.IGNORECASE):
        #     self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=True)
        # elif re.match(r"^retinanet$", name, re.IGNORECASE):
        #     self.model = models.detection.retinanet_resnet50_fpn(pretrained=True, progress=True)
        # elif re.match(r"^ssd300$", name, re.IGNORECASE):
        #     self.model = models.detection.ssd300_vgg16(pretrained=True, progress=True)
        # elif re.match(r"^ssdlite$", name, re.IGNORECASE):
        #     self.model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True, progress=True)
        # elif re.match(r"^m_resnet50$", name, re.IGNORECASE):
        #     self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)
        

        # elif re.match(r"^fcn_resnet50$", name, re.IGNORECASE):
        #     self.model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)
        # elif re.match(r"^fcn_resnet101$", name, re.IGNORECASE):
        #     self.model = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
        # elif re.match(r"^dl_resnet50$", name, re.IGNORECASE):
        #     self.model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        # elif re.match(r"^dl_resnet101$", name, re.IGNORECASE):
        #     self.model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        # elif re.match(r"^dl_mobile$", name, re.IGNORECASE):
        #     self.model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
        # elif re.match(r"^lraspp$", name, re.IGNORECASE):
        #     self.model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True, progress=True)
    
    
    def forward(self, x):
        return self.model(x)
    

def get_model(name: str):
    model = Model(name=name)
    model.eval()

    return model
