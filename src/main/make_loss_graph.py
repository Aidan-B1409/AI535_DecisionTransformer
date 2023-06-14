import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os 
import seaborn as sns

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True, help='Directory of json files')
    return parser.parse_args()


def main():
    args = parseargs()
    sns.set()
    dfs = {}
    for path in os.listdir(args.input):
        df = pd.read_json(os.path.join(args.input, path), orient='records')
        path_parts = path.split('.')[0]
        df_name = "".join(path.split('_')[2:-1])
        print(path_parts)
        print(df)
        df = df.set_index('epoch')
        dfs[df_name] = df

    data = pd.DataFrame(index=list(dfs.values())[0].index)
    for k, df in dfs.items():
        print(df)
        sns.lineplot(data=df, x='epoch', y='loss', label=k)

    # title
    # style ggplots
    # sns.lineplot(data=data, x='epoch', y='loss')
    plt.show()


if __name__ == '__main__':
    main()
