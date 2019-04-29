import torch
import torch.nn as nn


class SpatialStream(nn.Module):
    def __init__(self, num_classes, model, model_name):
        super(SpatialStream, self).__init__()

        if model_name is 'vgg16_bn':
            model.classifier[6] = nn.Linear(model.classifier[3].out_features, num_classes)

        self.features = model.features
        self.classifier = model.classifier

    def forward(self, x):
        frame_x = torch.tensor([])
        frame_conv13 = torch.tensor([])
        if self.model_name is 'vgg16_bn':
            for frame in range(len(x)):
                conv13 = self.features(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                torch.cat((frame_x, x))
                torch.cat((frame_conv13, conv13))

        return torch.mean(x, 0), frame_conv13
