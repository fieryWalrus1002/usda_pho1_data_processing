import pandas as pd
import glob
import os

def get_date_from_filename(filename):
    """ filenames will look like:
    062324_maize\\output_data\\20240623_062324_maize_HC69_5_DA_p_vals.csv
    062324_maize\\output_data\\20240623_062324_maize_HC69_5_DA_fluor.csv
    
    This function will extract the date from the filename, which is the first part of the filename.
    
    """
    base_filename = os.path.basename(filename)
    date = base_filename.split("_")[0]
    return date


def main():

    print(f"running main in {os.getcwd()}")
    files = glob.glob("output_data/*_fluor.csv")

    data_date = get_date_from_filename(files[0])

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

    df.to_csv(f"output_data/{data_date}_combined_fluor.csv", index=False)

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

    df.to_csv(f"output_data/{data_date}_combined_pvals.csv", index=False)

if __name__ == "__main__":
    main()