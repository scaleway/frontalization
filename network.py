import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


''' Generator network for 128x128 RGB images '''
class G(nn.Module):
    
    def __init__(self):
        super(G, self).__init__()
        
        self.main = nn.Sequential(
            # Input HxW = 128x128
            nn.Conv2d(3, 16, 4, 2, 1), # Output HxW = 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1), # Output HxW = 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), # Output HxW = 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # Output HxW = 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), # Output HxW = 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1), # Output HxW = 2x2
            nn.MaxPool2d((2,2)),
            # At this point, we arrive at our low D representation vector, which is 512 dimensional.

            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias = False), # Output HxW = 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # Output HxW = 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # Output HxW = 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False), # Output HxW = 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias = False), # Output HxW = 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias = False), # Output HxW = 128x128
            nn.Tanh()
        )

    
    def forward(self, input):
        output = self.main(input)
        return output


''' Discriminator network for 128x128 RGB images '''
class D(nn.Module):
    
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
                                  nn.Conv2d(3, 16, 4, 2, 1),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(16, 32, 4, 2, 1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(32, 64, 4, 2, 1),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(64, 128, 4, 2, 1),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(128, 256, 4, 2, 1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(256, 512, 4, 2, 1),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(512, 1, 4, 2, 1, bias = False),
                                  nn.Sigmoid()
                                  )
    
    
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)
