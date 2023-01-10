import os
import scipy.io
import numpy as np
import pandas as pd
import tifffile
import torch

# data reading functions. THEY SHOULD HAVE THE SAME ARGUMENT STYLE.
def read_matfile(device,filename,fileindex=""):
    """
    Read multispectral image(3 dimention) from matlab file and converts it to be ndarray in [C,H,W].
    (NOTICE: Input file format is expected to the same as the output of Tamamitsu-san's Photothermal program.)
    """
    #dafault index string for the image data in .mat file is matfile_name-".mat"(so, [:-4] in slice) if matfile_index parametor is not specified.
    image_index = filename[:-4] if (fileindex=="") else fileindex
    # Loading image data
    image_data = (scipy.io.loadmat(filename))[image_index]
    assert image_data.ndim == 3 , "Dimention of the input file does not match to the expected dimention(==3)."
    # Transpose data to be [C,H,M] 
    image_data = image_data.transpose(2,0,1)
    #Convert to Tensor on ${device} (pytorch expects float data to be 32 bits)
    image_data_tensor = torch.from_numpy(image_data.astype(np.float32)).to(device)
    return image_data_tensor

def read_tifffile(device,filename,fileindex=""):
    """
    ## read_tifffile(device,tifffile_name)
    Read tiff file and convert it to be adarray in [C,H,W]
    ### NOTICE
    Do not use "fileindex" argument.
    """
    # load tiff file
    image_data = tifffile.imread(filename)
    # image is already in the shape: [C,H,M] 
    #Convert to Tensor on ${device} (pytorch expects float data to be 32 bits)
    image_data_tensor = torch.from_numpy(image_data.astype(np.float32)).to(device)
    return image_data_tensor


class ImageDataset(torch.utils.data.Dataset):
    """
    ## class ImageDataset(torch._utils.data.Dataset)
    Modified dataset of Dataset in pytorch.

    ### TODO
    - enable buffering function
    - write info() code
    - write transformarion method --> DONE!
    """

    # For data transformation procedure.
    # Storing important parametor (using self.X = X) for prediction is recommended.
    ###################################################################################
    #                                   DATA TRANSFORMS                               #
    ###################################################################################
    # 1 Transform func
    def signal_transform(self, signal):
        result = signal
        self.mean_signal = torch.mean(signal,dtype=torch.float32)
        result -= self.mean_signal
        self.std_signal = torch.std(signal)
        result /= self.std_signal
        return result

    def target_transform(self, target):
        result = target
        self.mean_target = torch.mean(target,dtype=torch.float32)
        result -= self.mean_target
        self.std_target = torch.std(target)
        result /= self.std_target
        return result
    ################################################################################
    # 2 Inverse func
    # Name the inverse functions of the transform functions above as 
    # [Name of Transform func]_re(), respectively.
    def signal_transform_re(self, signal_re):
        result = signal_re
        result *= self.std_signal
        result += self.mean_signal
        return result
    
    def target_transform_re(self, target_re):
        result = target_re
        result *= self.std_target
        result += self.mean_target
        return result
    ###################################################################################
    ###################################################################################


    def __init__(self,device,data_csv, datareader=""):
        super().__init__()
        # The csv file thae lists the image files(default:.mat file)
        self.data_csv = data_csv
        self.device = device
        # Read the csv file and set some parametors
        self.df = pd.read_csv(self.data_csv)
        #check the column index of the csv file
        assert (self.df.columns.tolist() == ["signal_path", "target_path"])
        self.data_len = len(self.df.index)
        #data reader type (matfile or tifffile)
        self.datareader = datareader
        # set reader type
        if(self.datareader == "matfile"):
            self.reader = read_matfile
        elif(self.datareader == "tifffile"):
            self.reader = read_tifffile
        else:
            print("Reading function is not specified")
            return

    def __len__(self):
        return self.data_len

    def __getitem__(self,index):
        #signal file
        signalfile = self.df["signal_path"][index]
        signaltensor = self.reader(self.device,signalfile)
        #target file
        targetfile = self.df["target_path"][index]
        targettensor = self.reader(self.device,targetfile)
        #add channel in dim = 0 for the transforms
        targettensor = torch.unsqueeze(targettensor,dim=0)
        #transform
        signaltensor = self.signal_transform(signaltensor)
        targettensor = self.target_transform(targettensor)
        return  signaltensor,targettensor#torch.squeeze(targettensor,dim=0)

    def info(self):
        pass
        

class Data(object):
    """
    ## class Data(object)
    Class for loading and converting data from input files.
    #### [USAGE]:
    - STEP1) Make signal & target .mat files and .csv files (for training and validation).
    - STEP2) Initialize Data class with some parametor (including the information of the files made in STEP1) )
    - STEP3) Make Dataset using make_dataset() method
    - STEP4) Make Dataloader using make_dataloader() method

    ## Init Patameters
    #### FILE NAMES
    - learnfile_name : File name of the data used for learing
    - valfile_name : File name of the data used for validation
    #### OTHERS
    - PATH : PATH to data directory (optional)
    - device : Device input data is stored in (optional,default: "cpu")
    ## Memebers & methods
    """
    def __init__(self,learnfile_name:str ,valfile_name:str, datareader="tiffreader",device='cpu',batch_size=10):
        #If there not exist the input files, assert.
        assert os.path.exists(learnfile_name), "File does not exist."
        assert os.path.exists(valfile_name), "File does not exist."
        
        self.learnfile_name = learnfile_name
        self.valfile_name = valfile_name
        self.device = device
        self.batch_size = batch_size
        self.datareader = datareader

        #for storing instance of ImageDataset.
        self.image_dataset = None
        self.image_dataset_val = None
        self.image_dataloader = None
        self.image_dataloader_val = None

    def make_dataset(self):
        """
        ## make_dataset(data,device)
        Converts ndarray to tensor on specified device. ndarray is expected to be float.64.
        """
        self.image_dataset = ImageDataset(
            self.device, data_csv=self.learnfile_name, datareader=self.datareader)
            
        self.image_dataset_val = ImageDataset(
            self.device, data_csv=self.valfile_name, datareader=self.datareader)

        return

    def make_dataloader(self):
        # If Dataset is not set already:
        if(self.image_dataset == None or self.image_dataset_val == None):
            self.make_dataset(self)
        #for training
        self.image_dataloader = torch.utils.data.DataLoader(self.image_dataset,batch_size=self.batch_size,shuffle=True)
        #for validation
        self.image_dataloader_val = torch.utils.data.DataLoader(self.image_dataset,batch_size=self.batch_size,shuffle=True)
        return