import numpy as np
import argparse
from PIL import Image
import tempfile
import cv2
import image_similarity_measures
from image_similarity_measures.quality_metrics import fsim
import matplotlib.pyplot as plt

def PCC(x,y):
    """
    Calculate Pearson product-moment correlation coefficients of the
    two input 2D ndarrays:x and y.
    Firstly,convert 2D ndarray into 1D array, using "numpy.ndarray.ravel()
    And then, they are passed to np.corrcoef, and return.

    Args:
        x (ndarray): 2D ndarray must have the same size as y.
        y (ndarray): 2D ndarray must have the same size as x.

    Returns:
        num: Pearson product-moment correlation coefficient(PCC) between two matrises.
    """
    #Convert 2D ndarray into 1D array, using "numpy.ndarray.ravel()"
    #.ravel() returns 1D view of the input.
    #It does not allocate new memory,it returns just a view.
    pixel_data = [x.ravel(),y.ravel()]
    return np.corrcoef(pixel_data)[0,1]

def NRMSE(x,y):
    """
    Calculate NRMSE(normalized root-mean-square error) of the
    two input 2D ndarrays:x and y.
    I used y(true-image) for normalize RMSD, that is,
    RMSD is divided by (y.max()-y.min()).
    
    Args:
        x (ndarray): 2D ndarray must have the same size as y.
        y (ndarray): 2D ndarray must have the same size as x.

    Returns:
        num: NRMSE(normalized root-mean-square error) between two matrises.
    """
    RMSD = np.sqrt(np.mean((x-y)**2))
    NRMSD = RMSD/(y.max()-y.min())
    return NRMSD

def FSIM(x,y):
# USED CODE HERE:https://github.com/up42/image-similarity-measures
# DO NOT FORGET TO CITE.
    with tempfile.TemporaryDirectory() as dname:
        # do not show axis
        # save output tmp file.
        plt.axis("off")
        plt.imshow(x)
        plt.clim(20000, 65000)
        plt.savefig(dname+"/pred.png")
        plt.savefig("pred.png")

        plt.imshow(y)
        plt.clim(20000, 65000)
        plt.savefig(dname+"/ref.png")
        plt.savefig("ref.png")

        # Read saved imgs
        pred = cv2.imread(dname+"/pred.png")
        ref = cv2.imread(dname+"/ref.png")
        fsim_result = fsim(pred,ref)
    return fsim_result
def main():
    """
    ## TODO
    - Modify this tool so that this can read csv file that contains true-image npy files vs predicted-npy files information, 
    and calculate quality metrics sequentially.
    """
    #Making Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("predicted_image",help="PATH for the predicted image.(in the shape of .npy)")
    parser.add_argument("true_image",help="PATH for the true image.(in the shape of .npy)")
    parser.add_argument("-t", "--type",default="PCC",help="Specify which type of accuracy metric you want to use: PCC or NRMSE or FSIM or all. It is set to PCC by default. ")
    args = parser.parse_args()
    #Loading ndarray
    x = np.load(args.predicted_image, allow_pickle=True).astype(np.float32)
    y = np.load(args.true_image, allow_pickle=True).astype(np.float32)
    #Main part
    print(args.predicted_image+"  VS  "+args.true_image)
    print("-----------------------------------")
    if args.type=="PCC" or args.type=="all":
        result=PCC(x,y)
        print("PCC:",result)
    if args.type=="NRMSE" or args.type=="all":
        result=NRMSE(x,y)
        print("NRMSE:",result)
    if args.type=="FSIM" or args.type=="all":
        result=FSIM(x,y)
        print("FSIM:",result)
    else:
        print("You're wrongly specified -t or --type argument.")
        return
    return

if __name__ == '__main__':
    main()