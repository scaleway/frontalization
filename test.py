import torch
import torchvision.utils as vutils
from torch.autograd import Variable
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from data import ImagePipeline

device = 'cuda'

datapath = 'test_set'

# Generate frontal images from the test set
def frontalize(model, datapath, mtest):
    
    test_pipe = ImagePipeline(datapath, image_size=128, random_shuffle=False, batch_size=mtest)
    test_pipe.build()
    test_pipe_loader = DALIGenericIterator(test_pipe, ["profiles", "frontals"], mtest)
    
    with torch.no_grad():
        for data in test_pipe_loader:
            profile = data[0]['profiles']
            frontal = data[0]['frontals']
            generated = model(Variable(profile.type('torch.FloatTensor').to(device)))
    vutils.save_image(torch.cat((profile, generated.data, frontal)), 'output/test.jpg', nrow=mtest, padding=2, normalize=True)
    return

# Load a pre-trained Pytorch model
saved_model = torch.load("./output/netG_1.pt")

frontalize(saved_model, datapath, 3)

