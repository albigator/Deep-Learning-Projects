import torch.nn as nn
import torchvision.models as models

class FCN(nn.Module):

    def __init__(self, n_class, feature_extract=True):
        super().__init__()
       
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.relu2   = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)


    def forward(self, x):
        #x1 = __(self.relu(__(x)))
        # Complete the forward function for the rest of the encoder
        out_encoder = self.bnd5(self.conv5(self.bnd4(self.conv4(self.bnd3(self.conv3(self.bnd2(self.conv2(self.bnd1(self.conv1(x))))))))))
        #out_encoder = self.layer4(self.layer3(self.layer2(self.layer1(self.maxpool(self.bn1(self.conv1(x)))))))

        
        mid = self.relu2(out_encoder)
        out_decoder = self.bn5(self.deconv5(self.bn4(self.deconv4(self.bn3(self.deconv3(self.bn2(self.deconv2(self.bn1(self.deconv1(mid))))))))))
        # Complete the forward function for the rest of the decoder
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)