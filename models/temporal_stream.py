import torch
import torch.nn as nn


class TemporalStream(nn.Module):
    def __init__(self, model, model_name, frames_temporal_flow=10, num_classes=None):
        super(TemporalStream, self).__init__()

        # Currently tested for VGG. Other networks require adaptations
        # TODO: keep trained values for the first layer
        # print(model.state_dict()['features.0.weight'])
        # print(model.state_dict()['features.0.bias'])
        if model_name is 'vgg16_bn':
            out_channels = model.features[0].out_channels
            kernel_size = model.features[0].kernel_size
            stride = model.features[0].stride
            padding = model.features[0].padding
            model.features[0] = nn.Conv2d(frames_temporal_flow * 2, out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding)
            if num_classes:
                model.classifier[6] = nn.Linear(model.classifier[3].out_features, num_classes)

            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier

        self.model_name = model_name
        self.num_classes = num_classes
        self.frames_temporal_flow = frames_temporal_flow

    def forward(self, x):
        if self.model_name is 'vgg16_bn':
            res_mean = torch.zeros(x.size(0), self.num_classes).cuda()
            batch_size = x.size(1) // (self.frames_temporal_flow * 2)
            # print(x.size(1), self.frames_temporal_flow * 2, batch_size)
            for frame in range(batch_size):
                index = frame * self.frames_temporal_flow * 2
                res = self.features(x[:, index:index + self.frames_temporal_flow * 2, :, :])
                # Perform fusion at this point
                res = self.avgpool(res)
                res = res.view(res.size(0), -1)
                res = self.classifier(res)
                res_mean += res / batch_size

        return res_mean
