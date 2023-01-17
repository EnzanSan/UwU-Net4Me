import torch
import dataload as dl
from PIL import Image
import numpy as np
import tifffile
import matplotlib.cm as cm
import matplotlib.pyplot as plt

FILENAME="/home/toyama/UwU/csv/srs/A23_srs.tif"
REFFILENAME="/home/toyama/UwU/csv/fluo/A23_fluo.tif"
PTH_FILENAME="/home/toyama/UwU/[UwU]-4-8-10-8-6[batch:10-iter1000].pth"
SAVEFILENAME="/home/toyama/UwU/test2/A46"
def rec_ref(target:any, depth:int,names:list):
    """
    A tool for recursively accesssing members of classes.
    """
    result_obj = target
    for i in range(depth):
        for index in names:
            result_obj = vars(result_obj)[index]
    return result_obj

class seg_grad_cam:
    def __init__(self,model,target_layer,M,img_h = 512,img_w = 512):
        self.img_h = img_h
        self.img_w = img_w
        self.model = model
        self.target_layer = target_layer
        self.M = M
        # Turn the model into evaluation mode
        self.model.net.to('cpu')
        self.model.net.eval()
        # hooks list which will be removed in the end
        self.hooks = []

    def forward(self, x):
        return self.model.net(x)

    def reg_hooks(self):
        def save_feature_grad(module, in_grad, out_grad):
            self.feature_grad = out_grad[0]
        def save_feature_map(module, inp, outp):
            self.feature_map = outp[0]
        self.hooks.append(self.target_layer.register_backward_hook(save_feature_grad))
        self.hooks.append(self.target_layer.register_forward_hook(save_feature_map))

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
#print(model.net.net_recurse)
# input the model to class seg_grad_cam
FEATURE_LAYER = model.net.net_recurse.sub_u.sub_u.sub_2conv_more.relu1 # rec_ref(model.net.net_recurse.sub_u,1,['sub_u']).bottleneck
M = []
for i in range(256-25,256+25):
    for j in range(256-25,256+25):
        M.append((i,j))
segradcam = seg_grad_cam(model,FEATURE_LAYER,M,img_h=512,img_w=512)
segradcam.reg_hooks()

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

#intermediate value
forward = segradcam.feature_map
backward = segradcam.feature_grad.squeeze(0)

print(forward.shape)
print(backward.shape)
segradcam.clear_hook()

forward_np = forward.detach().numpy()
backward_np = backward.detach().numpy()

outp = np.zeros_like(backward_np[0,:,:])
for i in range(len(forward_np[:,0,0])):
    outp = np.add(outp,backward[i].mean()*forward_np[i])

plt.imshow(outp,cmap="jet")
plt.colorbar()
plt.savefig("out"+".png")

plt.imshow(forward_np[1,:,:],cmap="viridis")
plt.colorbar()
plt.savefig("map"+".png")

plt.imshow(backward_np[0,:,:],cmap="viridis")
plt.colorbar()
plt.savefig("grad"+".png")

## UwUの実装において、スペクトル学習の後のU-Netについて、スペクトル毎のfor文で回しているため、
## final_chanの数だけforwardとbackwardが発生しているものとおもわれる。
