import torch
import torch.nn as nn

import torchvision.models as torchvision_models

"""
Implementation of the Convolutional Two-Stream Network Fusion used for Video Action Recognition, by Feichtenhofer et al.
This network consists of two streams: temporal and spacial.
The temporal stream uses optical flow frames computed in advance. In the original paper the optical flow stream has a 
temporal receptive field of 10 frames. This stream has an adapted input convolutional layer with twice as many channels
as flow frames (because flow has a vertical and horizontal channel.)


"""


# The paper suggests multiple ways to perform the spatial fusion, but this is
class ConvFusionUnit(nn.Module):
    """
    First stacks the two feature maps at the same spatial locations and convolves the data with a bank of filters
    and biases.
    """
    def __init__(self, spatial_channels, temporal_channels):
        self.stacked_features = torch.cat(spatial_channels, temporal_channels)


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





class TwoStreamFusion(nn.Module):

    def __init__(self, num_classes=101, model='vgg16_bn', flow_frames=10):
        super(TwoStreamFusion, self).__init__()

        available_models = ['vgg16_bn']
        model_zoo = torchvision_models
        if model not in available_models:
            print("=> Model not found or supported. Using the baseline model VGG16 with BN")
            model = 'vgg16_bn'
            model_spatial = torchvision_models.vgg16_bn(pretrained=True)
            model_temporal = torchvision_models.vgg16_bn(pretrained=True)
        else:
            model_scatial = model_zoo.__dict__[model](pretrained=True)
            model_temporal = model_zoo.__dict__[model](pretrained=True)

        temporal_stream = TemporalStream(num_classes, flow_frames, model_temporal, model)




    def forward(self, rgb, flow):
        pass

    def get_model_names(self, models):
        return [name for name in models.__dict__ if name.islower() and not name.startswith('__')
                and callable(models.__dict__[name])]

TwoStreamFusion()