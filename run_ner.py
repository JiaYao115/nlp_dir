from models_ner import albert
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

################ args
parse = argparse.ArgumentParser()
parse.add_argument('--model', default='albert')
parse.add_argument('--training', default=True, type=bool)
parse.add_argument('--data_path', default='./data')
args = parse.parse_args()
print(args)


################ data
config = albert.Config()
dataset_train = []
dataset_test = []
dataset_val = []
dataLoader_train = DataLoader(dataset_train, config.batch_size)
dataLoader_test = DataLoader(dataset_test, config.batch_size)
dataLoader_val = DataLoader(dataset_val, config.batch_size)
print(len(dataLoader_train))
print(len(dataLoader_test))
print(len(dataLoader_val))

################ model

model = albert.Module(config)


if __name__ == '__main__':
    pass
