import torch.nn as nn


class TemporalStream(nn.Module):
    def __init__(self, num_classes, flow_frames, model, model_name):
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
            model.features[0] = nn.Conv2d(flow_frames * 2, out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding)
            model.classifier[6] = nn.Linear(model.classifier[3].out_features, num_classes)

            self.features = model.features
            self.classifier = model.classifier

        self.model_name = model_name
        print(self.features, self.classifier)