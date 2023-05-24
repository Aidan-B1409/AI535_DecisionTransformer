import pickle
import argparse
import numpy as np
import pandas as pd

def main():
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
    # print(objects[0])
    # df = pd.DataFrame.from_dict(objects[0])
    # print(df)
    # print(df.summary())

if __name__ == '__main__':
    main()
