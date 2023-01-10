import torch
import importlib

#For initialization of the model weight.
# For the reason why this initialization method is needed, refer to the original U-Net and UwU-Net paper.
#From [https://github.com/B-Manifold/pytorch_fnet_UwUnet/blob/master/fnet/fnet_model.py]
###################################################
def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 
##################################################

class Model(object):
    def __init__(self,dataloader, module_name="module.UwU", init_weights = True, lr = 0.001, criterion_fn = torch.nn.MSELoss, module_kwargs={},device="cpu"):
        #Basic information about the model
        self.module_name = module_name
        self.init_weights = init_weights
        self.lr = lr
        self.criterion_fn = criterion_fn()
        self.module_kwargs = module_kwargs
        self.device = device
        self.iter_times = 0
        # Dataset and Dataloader
        self.dataloader = dataloader

        #Initialization of the model
        self.net = importlib.import_module(self.module_name).Net(**module_kwargs)

        #weight initialization. (Why this make sense? --> refer to the U-Net paper & UwU-Net papar)
        if self.init_weights:
            self.net.apply(_weights_init)

        #deploy the model to the specified device
        self.net.to(device)

        #setting of the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def train(self, signal, target):
        self.net.train()
        self.optimizer.zero_grad()
        output = self.net(signal)
        loss = self.criterion_fn(output, target)
        loss.backward()
        self.optimizer.step()
        self.iter_times += 1
        return loss.item()
        
    def predict(self, signal):
        self.net.eval()
        pred = self.net(signal)
        return pred

    def savemodel(self,savefilename):
        self.savefilename = savefilename
        torch.save(self,"["+self.module_name+"]"+self.savefilename)
            