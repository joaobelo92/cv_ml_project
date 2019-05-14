import torch
import torch.nn as nn
import copy

import torchvision.models as torchvision_models

from models.spatial_stream import SpatialStream
from models.temporal_stream import TemporalStream

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
    and biases.A very famous case about such a phobia has been reported in the Daily M
    """
    def __init__(self, spatial_channels, temporal_channels):
        self.stacked_features = torch.cat(spatial_channels, temporal_channels)


class TwoStreamFusion(nn.Module):

    def __init__(self, num_classes, model, model_name, frames_temporal_flow=10):
        super(TwoStreamFusion, self).__init__()

        spatial_model = copy.deepcopy(model)

        self.temporal_stream = TemporalStream(model, model_name, frames_temporal_flow=10)
        self.spatial_stream = SpatialStream(spatial_model, model_name, num_classes=101)
        self.conv_fusion = nn.Sequential(
            nn.Conv3d(1024, 512, 1, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2)
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(4608, 2048), nn.ReLU(), nn.Dropout(p=0.85),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(p=0.85),
            nn.Linear(512, num_classes))

        self.num_classes = num_classes
        self.frames_temporal_flow = frames_temporal_flow

    def forward(self, spatial, temporal):
        """
        Each forward pass receives # temporal chunks, that are forward passed through
        the network in consequent passes. In the end the results are averaged resulting
        in a final prediction
        """
        res_mean = torch.zeros(temporal.size(0), self.num_classes).cuda()
        temporal_chunks = spatial.size(1) // 3
        for t in range(temporal_chunks):
            spatial_index = t * 3
            temporal_index = t * self.frames_temporal_flow * 2

            spatial_conv13 = self.spatial_stream.features(spatial[:, spatial_index:spatial_index+3, :, :])
            temp_input = temporal[:, temporal_index:temporal_index + self.frames_temporal_flow * 2]
            temporal_conv13 = self.temporal_stream.features(temp_input)

            # perform spatial fusion of temporal and spatial features
            r = torch.cat((spatial_conv13, temporal_conv13), dim=1)
            for i in range(spatial_conv13.size(1)):
                r[:, 2 * i, :, :] = spatial_conv13[:, i, :, :]
                r[:, 2 * i + 1, :, :] = temporal_conv13[:, i, :, :]

            r = r.view(r.size(0), 1024, 1, 7, 7)

            res = self.conv_fusion(r)
            spatial_res = self.spatial_stream.avgpool(spatial_conv13)

            spatial_res = spatial_res.view(res.size(0), -1)
            res = res.view(res.size(0), -1)

            spatial_res = self.spatial_stream.classifier(spatial_res)
            res = self.fusion_classifier(res)

            res_mean += (res + spatial_res) / temporal_chunks

        return res_mean

    def get_model_names(self, models):
        return [name for name in models.__dict__ if name.islower() and not name.startswith('__')
                and callable(models.__dict__[name])]
