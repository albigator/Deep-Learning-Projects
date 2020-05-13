import torch.nn as nn

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(16)
        self.conv2   = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(32)
        self.relu1   = nn.ReLU()
        self.conv3   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(64)
        self.relu2   = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(32)
        self.relu3   = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        #x1 = __(self.relu(__(x)))
        # Complete the forward function for the rest of the encoder
        out_encoder = self.relu2(self.bnd3(self.conv3(self.relu1(self.bnd2(self.conv2(self.bnd1(self.conv1(x))))))))
        mid = self.relu2(out_encoder)
        out_decoder = self.bn3(self.deconv3(self.relu3(self.bn2(self.deconv2(self.bn1(self.deconv1(mid)))))))
        # Complete the forward function for the rest of the decoder
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)