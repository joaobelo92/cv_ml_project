import torch
import torch.nn as nn

class TemporalStream(nn.Module):
    def __init__(self, model, model_name, frames_temporal_flow=10, num_classes=None):
        super(TemporalStream, self).__init__()

        # Currently tested for VGG. Other networks require adaptations
        if model_name == 'vgg16_bn':
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
        elif model_name == 'resnet18':
            out_channels = model.conv1.out_channels
            kernel_size = model.conv1.kernel_size
            stride = model.conv1.stride
            padding = model.conv1.padding
            conv1 = nn.Conv2d(frames_temporal_flow * 2, out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding)
            self.features = nn.Sequential(
                conv1,
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
        elif model_name == 'shufflenetv2_1' or model_name == 'shufflenetv2_2':
            out_channels = model.conv1.out_channels
            kernel_size = model.conv1.kernel_size
            stride = model.conv1.stride
            padding = model.conv1.padding
            conv1 = nn.Conv2d(frames_temporal_flow * 2, out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding)
            self.features = nn.Sequential(
                conv1,
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
        self.frames_temporal_flow = frames_temporal_flow

    def forward(self, x):
        res_mean = torch.zeros(x.size(0), self.num_classes).cuda()
        batch_size = x.size(1) // (self.frames_temporal_flow * 2)
        # print(x.size(1), self.frames_temporal_flow * 2, batch_size)
        for frame in range(batch_size):
            index = frame * self.frames_temporal_flow * 2
            curr_input = x[:, index:index + self.frames_temporal_flow * 2, :, :]
            res = self.features(curr_input)
            # Perform fusion at this point
            res = self.avgpool(res)
            res = res.view(res.size(0), -1)
            res = self.classifier(res)
            res_mean += res / batch_size

        return res_mean
