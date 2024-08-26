# gather all of the data files in all subdirectories into the ./collected_data directory
# D:\data\usda_pho1_data_processing\072524_rice\output_data\20240725_combined_fluor.csv
# D:\data\usda_pho1_data_processing\072524_rice\output_data\20240725_combined_pvals.csv
# and so on

import os
import glob


def main():
    
    # get all of the data files in the subdirectories
    data_files = glob.glob('**/output_data/*combined*.csv', recursive=True)

    print(len(data_files))
        
    # create the collected_data directory if it doesn't exist
    if not os.path.exists('collected_data'):
        os.makedirs('collected_data')
    
    # copy all of the data files to the collected_data directory
    for file in data_files:
        os.system(f'copy {file} collected_data')


if __name__ == "__main__":
    main()
    