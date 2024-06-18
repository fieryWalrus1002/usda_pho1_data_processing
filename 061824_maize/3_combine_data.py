import pandas as pd
import glob
import os

if __name__ == "__main__":

    print(f"running main in {os.getcwd()}")
    files = glob.glob("output_data/*_fluor.csv")

    dfs = []

    for filename in files:
        print(filename)

        dfs.append(pd.read_csv(filename))

    df = pd.concat(dfs)

    df.drop(columns=["Unnamed: 0", "reference"], inplace=True)

    # reset the index
    df.reset_index(drop=True, inplace=True)

    # change the column order, so that the last three columns are the first three
    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]

    df = df[cols]

    df.to_csv("output_data/040324_combined_fluor.csv", index=False)

    # combine the pvals
    files = glob.glob("output_data/*_p_vals.csv")

    dfs = []

    for filename in files:

        dfs.append(pd.read_csv(filename))

    df = pd.concat(dfs)

    df.drop(columns=["Unnamed: 0"], inplace=True)

    # change the column order, so that the last three columns are the first three
    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]

    df = df[cols]

    # reset the index
    df.reset_index(drop=True, inplace=True)

    df.to_csv("output_data/040324_combined_pvals.csv", index=False)
