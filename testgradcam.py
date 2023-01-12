import torch
import dataload as dl
from PIL import Image
import numpy as np
import tifffile
import matplotlib.cm as cm
import matplotlib.pyplot as plt

FILENAME="/home/toyama/UwU/csv/srs/A46_srs.tif"
REFFILENAME="/home/toyama/UwU/csv/fluo/A46_fluo.tif"
PTH_FILENAME="[UwU]-4-6-10-8-6[batch:4-iter1].pth"
SAVEFILENAME="/home/toyama/UwU/test2/A46"

class seg_grad_cam:
    def __init__(self,model,feature_layer,M,img_h = 512,img_w = 512,clss = 1):
        self.feature_grad = []
        self.feature_map = []
        self.img_h = img_h
        self.img_w = img_w
        self.cls = clss
        self.model = model
        self.feature_layer = feature_layer
        self.M = M
        # Turn the model into evaluation mode
        self.model.net.to('cpu')
        self.model.net.eval()
        # hooks list which will be removed in the end
        self.hooks = []

    # make hooks
        def save_feature_grad(module, in_grad, out_grad):
            print(out_grad[0].shape)
            self.feature_grad.append(out_grad[0].detach())
        self.hooks.append(self.feature_layer.register_backward_hook(save_feature_grad))

        def save_feature_map(module, inp, outp):
            print(outp[0].shape)
            self.feature_map.append(outp[0].detach())
        self.hooks.append(self.feature_layer.register_forward_hook(save_feature_map))
    
    def forward(self, x):
        return self.model.net(x)

    def backward_prop(self,output):
        self.model.net.zero_grad()
        one_hot_output = torch.zeros([self.img_h,self.img_w])
        for pix in self.M:
            # one-hot encoding for the target pixels
            one_hot_output[pix] = 1.0
        output.backward(gradient=one_hot_output, retain_graph=True)
    
    def clear_hook(self):
        for hook in self.hooks:
            hook.remove()

#Load the trained model
model = torch.load(PTH_FILENAME)
print(model.net)
# input the model to class seg_grad_cam
FEATURE_LAYER = model.net.net_recurse.sub_2conv_more.relu1
M = []
for i in range(128,512-128):
    for j in range(128,512-128):
        M.append((i,j))
segradcam = seg_grad_cam(model,FEATURE_LAYER,M,img_h=512,img_w=512,clss=1)

#Load tif image and convert it into tensor form.(Using read_tifffile method in dataload.py)
input_image = dl.read_tifffile(device='cpu', filename=FILENAME)
input_image = torch.unsqueeze(input_image,dim=0)
# Transform
input_image = model.dataloader.image_dataset.signal_transform(input_image)
# forward prpergate
outputimg = segradcam.forward(input_image)
outputimg = torch.squeeze(outputimg,dim=0)
outputimg = torch.squeeze(outputimg,dim=0)
# back propargate
segradcam.backward_prop(outputimg)

#plt.imshow(segradcam.feature_grad[5].detach().numpy()[0,3,:,:],cmap="viridis")
#plt.colorbar()
#plt.savefig("grad.png")

#plt.clf()
#plt.imshow(segradcam.feature_map[3].detach().numpy()[3,:,:],cmap="viridis")
#plt.colorbar()
#plt.savefig("map.png")

segradcam.clear_hook()
## UwUの実装において、スペクトル学習の後のU-Netについて、スペクトル毎のfor文で回しているため、
## final_chanの数だけforwardとbackwardが発生しているものとおもわれる。
