import torch
from tqdm import tqdm
import argparse
import dataload as de
import basemodel as bm

#######################################
#UWU PARAMETORS.  EDIT HERE
#######################################
##Make heatmap for depth:1-8, mult_chan:1-4
params_list=[]
for dep in range(1,8+1):
    for mulc in range(1,4+1):
        params_list.append(
        {
        "mult_chan" : mulc,
        "depth" : dep,
        "starting_chan" : 10,
        "intermediate_chan" : 8,
        "final_chan" : 6
        } 
        )
#######################################
#######################################
params_list=[
    {
        "mult_chan" : 4,
        "depth" : 6,
        "starting_chan" : 10,
        "intermediate_chan" : 8,
        "final_chan" : 6
        } 
]

def main():
    #Making Parser
    #If you need help, try: training.py --help
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_train",help="Specify the csv file that contains signal vs target PATH pairs.These pairs are used for training.")
    parser.add_argument("--csv_val",help="Specify the csv file which contains signal vs target PATH pairs.These pairs are used for validation.")
    parser.add_argument("--datareader",help="Input file type. TIFF image file or .mat file.")
    parser.add_argument("--device",help="The device in which dataloader will be stored in.")
    parser.add_argument("--batch_size",type=int,help="Batch size.")
    parser.add_argument("--module",help="The module you are using for training.")
    parser.add_argument('--init_weights',action='store_true') 
    parser.add_argument('--lr',type=float,help='Learning rate') 
    parser.add_argument("--iter",type=int,help="Iteration number")
    args = parser.parse_args()

    # Initialize seed
    torch.manual_seed(317)
    #Making dataset
    data= de.Data(args.csv_train,args.csv_val,datareader=args.datareader,device=args.device,batch_size=args.batch_size)
    data.make_dataset()
    #Making dataloader
    data.make_dataloader()
    #making a NN model and training.
    #You can modify UwU-Net settings by changing module_kwargs.
    for params in params_list:
        model = bm.Model(data,module_name=args.module, init_weights = args.init_weights, lr = args.lr, criterion_fn = torch.nn.MSELoss,
        module_kwargs=params,device=args.device)
        for k in tqdm(range(args.iter)):
            for i,j in data.image_dataloader:
                model.train(i,j)
        #trained model is saved as 
        #[model]-mult_chan-depth-starting_chan-intermediate_chan-final_chan+"[batch:"+str(args.batch_size)+"-iter"+str(args.iter)+"]".pth
        NAME = ""
        for n in params.values():
            NAME = NAME + "-" + str(n)
        model.savemodel(NAME+"[batch:"+str(args.batch_size)+"-iter"+str(args.iter)+"]"+".pth")


if __name__ == '__main__':
    main()