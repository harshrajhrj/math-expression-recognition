import torch.nn as nn
from collections import OrderedDict

class MathSolverModel(nn.Module):
    def __init__(self, num_classes):
        super(MathSolverModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(2048, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
        self.initialize_weights()

    def forward(self, x):
        x = self.cnn(x)  # (B, C, H, W)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H)
        x = x.view(b, w, c * h)  # (B, W, C*H)
        x, _ = self.rnn(x)  # (B, W, 512)
        x = self.fc(x)  # (B, W, num_classes)
        x = x.permute(1, 0, 2)  # CTC expects (W, B, C)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
