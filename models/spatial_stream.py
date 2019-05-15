import torch
import torch.nn as nn


class SpatialStream(nn.Module):
    def __init__(self, model, model_name, num_classes=None):
        super(SpatialStream, self).__init__()

        if model_name == 'vgg16_bn' and num_classes:
            model.classifier[6] = nn.Linear(model.classifier[3].out_features, num_classes)
            self.features = model.features
            self.classifier = model.classifier
            self.avgpool = model.avgpool
        elif model_name == 'resnet18':
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4
            )
            self.avgpool = model.avgpool
            self.classifier = nn.Linear(512, num_classes)
        elif model_name == 'shufflenetv2_x2_0':
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.activation,
                model.max_pool,
                model.layer2,
                model.layer3,
                model.layer4,
                model.conv5
            )
            self.avgpool = model.avg_pool
            self.classifier = nn.Linear(model.out_channels[4], num_classes)

        self.model_name = model_name
        self.num_classes = num_classes

    def forward(self, x):
        res_mean = torch.zeros(x.size(0), self.num_classes).cuda()
        for frame in range(x.size(1) // 3):
            index = frame * 3
            conv13 = self.features(x[:, index:index+3, :, :])
            res = self.avgpool(conv13)
            res = res.view(res.size(0), -1)
            res = self.classifier(res)
            res_mean += res / (x.size(1) // 3)

        return res_mean

