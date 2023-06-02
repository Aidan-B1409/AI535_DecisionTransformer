import pickle
import argparse
import numpy as np
import pyarrow as pa
import pandas as pd
import datasets
from datasets import load_dataset

def parseargs():
    parser = argparse.ArgumentParser(help="")

def main():
    ds = dataloader()

def dataloader() -> datasets.Dataset:
    objects = []
    with open('dataset/expert/FetchPick/buffer.pkl', 'rb') as expert_fetch_pick:
        while True:
            try:
                objects.append(pickle.load(expert_fetch_pick))
            except EOFError:
                break
    print(objects)
    # Convert to np arrays
    for key in objects[0].keys():
        objects[0][key] = np.asarray(objects[0][key])
        print(objects[0][key].shape)
    print(objects[0]['ag'])

    ds = datasets.Dataset.from_dict(objects[0])
    for old, new in zip(ds.column_names, ['obs', 'actions', 'goal', 'achieved_goal']):
        ds = ds.rename_column(old, new)
    print(ds)

if __name__ == '__main__':
    dataloader()
