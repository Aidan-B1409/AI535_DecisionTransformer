import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True)
    return parser.parse_args()


def main():
    args = parseargs()
    df = pd.read_json(args.input, orient='records')
    print(df)

if __name__ == '__main__':
    main()
