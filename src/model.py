import torch 
import torch.nn as nn 
import torch.nn.functional as F

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes = 10):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding = 1),
#             nn.ReLU(), 
#             nn.MaxPool2d(2), 

#             nn.Conv2d(32, 64, kernel_size=3, padding = 1), 
#             nn.ReLU(), 
#             nn.MaxPool2d(2)
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(), 
#             nn.Linear(64 * 8 * 8, 128), 
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x 
    

# def get_model(config):
#     model_name = config["model"]["name"]
#     num_classes = config["model"]["num_classes"]

#     if model_name == "vanilla_cnn":
#         return(SimpleCNN(num_classes=10))
    
#     raise ValueError(f"Unknown Model Name: {model_name}")


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,out_channels, 3,1, 1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  

        if stride != 1 or in_channels != out_channels: # addition identity block special case when sizes  are different, and will cause problem during addition. 
            # Keeping kernel and 2*padding difference = 1 so that I do not need to check shape because of them 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out+= self.shortcut(x) # alwys added 
        out = F.relu(out)
        return out
        
        
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride = 2)

        self.linear = nn.Linear(512, num_classes)


    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def get_model(config):
    return ResNet([2,2,2,2], num_classes=10)
