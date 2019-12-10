import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class KeplerDataset(Dataset):
 
	def __init__(self, mode="train", transform=None):
		
		#Check whether train or test mode is selected
		if mode == "train":
			self.df = pd.read_csv('./data/exoTrain.csv')
		elif mode == "test":
			self.df = pd.read_csv('./data/exoTest.csv')
		self.truncate_length = 1500

	def __len__(self):
		return len(self.df)

	def normalize(self, X):
		#Normalize The Flux Values
		return (X - X.mean(axis=0))/X.std(axis=0)

	def __getitem__(self, idx):

		#Returns the flux time-series 
		X = torch.Tensor(self.df.iloc[idx][1 + self.truncate_length:-self.truncate_length])

		X = self.normalize(X)
		
		#Returns the label
		y = torch.Tensor([self.df.iloc[idx][0]]) - torch.Tensor([1])


		return X, y