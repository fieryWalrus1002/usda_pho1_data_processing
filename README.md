# usda_pho1_data_processing

Public repo for sharing the Python code to export data from the PhotoSynQ Multispeq JSON files, and plot the important bits.

## Instructions

1. **Create a Subfolder for Measurements**
    - Create a subfolder for the date of your measurement.
    - Place your JSON files into the `data` subfolder of that date.

2. **Copy and Run Python Scripts**
    - Copy over the three numbered Python scripts from the 'src' folder to your data directory and run them in order:
      1. **clear_dirs.py**: Clears the output folders and then creates them.
      2. **export_all.py**: Extracts all individual data files and processes them.
      3. **combine_data.py**: Combines individual data files into one data file for easier analysis.

3. **Output and Plots**
    - Plots and other outputs of the individual data are found in the various output folders, under the appropriate date subfolder.

4. **Naming Convention**
    - Follow the naming convention for data files as specified in the `export_all.py` script.

5. **Jupyter Notebook**
    - A Jupyter notebook is provided for developing scripts and analyzing individual data files.
    - Use it to test different parameters for the data processing functions and modify the `export_all` script as needed.

## Setting Up a Virtual Environment

To ensure a consistent development environment, it's recommended to use a virtual environment. Below are the steps to create and activate a virtual environment, as well as install the necessary packages.

### Step-by-Step Instructions

1. **Create and Activate the Virtual Environment**

   Open a terminal and navigate to the directory containing your `requirements.txt` file. Then run the following commands:

   ```sh
   # Create a virtual environment named 'env'
   python3 -m venv env

   # Activate the virtual environment
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```

2. ** Install the Required Packages **

	With the virtual environment activated, install the packages from the requirements.txt file:

   ```sh
    pip install -r requirements.txt
   ```

By following these steps, you'll have a virtual environment set up with all the required packages for this project.


