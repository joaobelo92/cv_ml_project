import torch
import torch.nn as nn


class SpatialStream(nn.Module):
    def __init__(self, num_classes, model, model_name):
        super(SpatialStream, self).__init__()

        if model_name is 'vgg16_bn':
            model.classifier[6] = nn.Linear(model.classifier[3].out_features, num_classes)

        self.features = model.features
        self.classifier = model.classifier
        self.avgpool = model.avgpool
        self.model_name = model_name

        print(model)

    def forward(self, x):
        if self.model_name is 'vgg16_bn':
            for frame in range(x.size(1) // 3):
                index = frame * 3
                print(x.size())
                conv13 = self.features(x[:, index:index+3, :, :])
                res = self.avgpool(conv13)
                res = res.view(res.size(0), -1)
                res = self.classifier(res)
                print(res.size())
                all_res = res if frame is 0 else torch.cat((all_res, res))
                frame_conv13 = conv13 if frame is 0 else torch.cat((frame_conv13, conv13))

        print(all_res.size())
        print(torch.mean(all_res, 0).size())
        return torch.mean(all_res, 0), frame_conv13

        # conv13 = self.features(x)
        # x = self.avgpool(conv13)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        # return x, conv13
