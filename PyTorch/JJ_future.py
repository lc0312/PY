import pandas as pd
import torch
from torch.autograd import Variable

data_raw =  pd.DataFrame (pd.read_csv ('D:\HW4_formulation data-1.xlsx')).to_numpy()
price = data_raw[:,1]

print (price)