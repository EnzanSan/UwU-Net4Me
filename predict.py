import torch
import dataload as dl
from PIL import Image
import numpy as np
import tifffile
import matplotlib.cm as cm
import matplotlib.pyplot as plt

FILENAME="/home/toyama/UwU/csv/srs/A46_srs.tif"
REFFILENAME="/home/toyama/UwU/csv/fluo/A46_fluo.tif"
PTH_FILENAME="[UwU]-4-6-10-8-6[batch:10-iter1000].pth"
SAVEFILENAME="/home/toyama/UwU/test2/A46"
#Load trained model
model = torch.load(PTH_FILENAME)
#Load tif image and convert it into tensor form.(Using read_tifffile method in dataload.py)
input_image = dl.read_tifffile(device=model.device, filename=FILENAME)
input_image = torch.unsqueeze(input_image,dim=0)
# Transform
input_image = model.dataloader.image_dataset.signal_transform(input_image)
# predict
result = model.predict(input_image)
# Transform
result = model.dataloader.image_dataset.target_transform_re(result)
# Convert a tensor to a numpy
result = torch.squeeze(result,dim=0)
result = torch.squeeze(result,dim=0)
result_np = result.to('cpu').detach().numpy().copy()

##################################
# SHOW THE RESULT TENSOR
#np.set_printoptions(threshold=np.inf)
#print(result_np)

# SAVE RESULT IMAGE
plt.imshow(result_np,cmap="viridis")
plt.clim(20000, 65000)
plt.colorbar()
plt.savefig(SAVEFILENAME+".png")

# SAVE RESULT NDARRAY
np.save(SAVEFILENAME,result_np)
# SAVE NDARRAY OF REFERENCE DATA
input_image_np = tifffile.imread(REFFILENAME)
np.save(SAVEFILENAME+"REF",input_image_np)
###################################