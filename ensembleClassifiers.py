# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv('train_csv')
    clf = RandomForestClassifier(n=4)


if __name__ == "__main__":
    main()
