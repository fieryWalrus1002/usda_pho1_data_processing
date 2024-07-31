import json
import os
import glob
import re
from datetime import datetime
import pytz
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# dict for translating the light number to a string
light_dict = {
    0: "dark",
    1: "530nm",
    2: "655nm",
    3: "590nm",
    4: "448nm",
    5: "950nm",
    6: "950nm",
    7: "655nm",
    8: "850nm",
    9: "730nm",
    10: "820nm",
}

# ## Define the functions we will use


def calc_linear_regression(x: np.array, y: np.array) -> Tuple[float, float]:
    # Create a DataFrame from the input arrays
    df = pd.DataFrame({"time_s": x, "dAbs": y})

    # Apply a rolling mean to the 'dAbs' column
    window_size = 5
    df["dAbs"] = df["dAbs"].rolling(window=window_size, center=True).mean()

    # Drop rows with NaN values
    df = df.dropna()

    # Fit a linear regression model
    lm = LinearRegression().fit(df["time_s"].values.reshape(-1, 1), df["dAbs"].values)

    # Extract the slope and intercept
    slope = lm.coef_[0]
    intercept = lm.intercept_

    return slope, intercept


def calc_delta_absorbance(transmission: np.array, baseline_value: float) -> np.array:
    """Calculates delta absorbance for p700 data using a given baseline."""
    return (transmission / baseline_value) / (-1 / 2.3)


# Function to calculate baseline for a specific label
def calculate_baseline_for_label(
    data: pd.DataFrame, begin: int = -50, end: int = -1
) -> float:
    """Calculates the baseline absorbance for a given label."""
    transmission = data["value"].values
    baseline_value = np.mean(transmission[begin:end])
    return baseline_value

def calculate_fvfm(data, pulse_start, pulse_end):
    # Calculate Fv/Fm
    # Fv = Fm - Fo
    # Fm = max fluorescence
    # Fo = minimal fluorescence
    # Fv/Fm = (Fm - Fo) / Fm    # Fo is the minimal fluorescence measured in the dark-adapted state

    # halfway through the pulse,
    pulse_midpoint = pulse_start + ((pulse_end - pulse_start) // 2)
    prepulse_midpoint = pulse_start // 2

    # reindex the data
    data = data.copy().reset_index(drop=True)

    Fo = data.iloc[pulse_start - 5 : pulse_start]["value"].mean()
    Fm = data.iloc[pulse_midpoint : pulse_end - 5]["value"].mean()
    FvFm = round((Fm - Fo) / Fm, 2)
    print(f"Fo: {Fo}, Fm: {Fm}, Fv/Fm: {FvFm}")

    return round((Fm - Fo) / Fm, 2)  # two decimal places is fine

# def calculate_fvfm(data, pulse_start, pulse_end):
#     # halfway through the pulse,
#     pulse_midpoint = pulse_start + ((pulse_end - pulse_start) // 2)
#     prepulse_midpoint = pulse_start // 2

#     # reindex the data
#     data = data.copy().reset_index(drop=True)

#     Fo = data.iloc[prepulse_midpoint : pulse_start - 5]["value"].mean()
#     Fm = data.iloc[pulse_midpoint : pulse_end - 5]["value"].mean()
#     FvFm = round((Fm - Fo) / Fm, 2)

#     return round((Fm - Fo) / Fm, 2)  # two decimal places is fine

def get_experiment_name(filename: str) -> str:
    """
    Take an experiment datafile and parse the experiment name from the file metadata
    into a string in the form of:
    '%Y%M%D_experiment_name'
    If experiment = testMacro, name = 012424_leaf_fr150_fr0SP, and created_at = 2024-01-25T00:58:36.782Z
    then the function should return:
    '20240125_testMacro_012424_leaf_fr150_fr0SP'
    """

    with open(filename) as file:
        obj = json.load(file)[0]

    experiment = obj["experiment"]
    name = re.sub(r"\d{6}(_)+", "", obj["name"])
    created_at = obj["created_at"]

    # Convert the timestamp to datetime object
    dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")

    # Set the timezone to UTC
    dt = dt.replace(tzinfo=pytz.UTC)

    # Convert the datetime object to PST
    dt_pst = dt.astimezone(pytz.timezone("US/Pacific"))

    # Format the datetime object as a string
    created_at_pst = dt_pst.strftime("%Y%m%d")

    print(f"experiment: {experiment}, name: {name}, created_at: {created_at_pst}")

    return created_at_pst + "_" + experiment + "_" + name


#### parse the varray into a dict for reference, with the various codes of the varray reference
### returning the values they represent when we query the varray.
### Eg varray["@n4:0"] = "820nm"
def parse_varray(vArrayDict) -> Dict:
    """dict, extract the parts we need and parse them into a dict for reference"""
    parsed = {}
    for i in range(len(vArrayDict)):
        for j in range(len(vArrayDict[i])):
            parsed[f"@n{i}:{j}"] = vArrayDict[i][j]
    return parsed


def get_data_from_file(file_path: str) -> Dict:
    print(f"Attempting to load data from : {file_path}")
    with open(file_path) as file:
        root = json.load(file)

    ########################## get the data set ###############################
    # does the data_set exist? check to see if it is nested in data, or just exposed as sample
    try:
        data_set = root[0]["data"]["sample"][0]["set"]
    except KeyError:
        try:
            data_set = root[0]["sample"][0]["set"]
        except KeyError:
            data_set = "blank"

    if data_set == "blank":
        print("no data set found")
    else:
        print("Data found, returning data_set with length: ", len(data_set))

    # sort data_set by time
    data_set = sorted(data_set, key=lambda k: k["time"])

    return data_set


def get_protocol_from_file(file_path: str) -> Dict:
    print(f"Attempting to load protocol from : {file_path}")
    with open(file_path) as file:
        root = json.load(file)

    protocolStr = root[0]["protocol"]

    try:
        # parse the json string into a dict
        protocol_dict = json.loads(protocolStr)[0]
    except TypeError:
        print("protocol is not a string, skipping")

    return protocol_dict


def parse_vlist(vlist: List[str], varray: dict) -> List[int]:
    parsed = []
    for item in vlist:
        if type(item) != str:
            parsed.append(int(item))
        else:
            parsed.append(int(varray[item]))
    return parsed


def getProtocolForLabel(label: str, protocols: dict) -> dict:
    protocolList = protocols["_protocol_set_"]
    for protocol in protocolList:
        if protocol["label"] == label:
            return protocol
    return None


# extract all of the start times for each trace in the protocol script, and calculate
# the time in milliseconds from the beginning of the experiment
# This allows us to create an accurate time series for the data
def get_experiment_start_times_ms(filename: str) -> dict:
    with open(filename) as file:
        root = json.load(file)[0]

    # The object may be nested in a "data" object, or just exposed as "sample"
    try:
        data = root["data"]
    except KeyError:
        data = root

    experiment_start_time = int(data["time"])
    time_dict = {"experiment_start": experiment_start_time}
    time_list = []

    for i, protocol in enumerate(data["sample"][0]["set"]):
        time_list.append(
            {
                "label": protocol["label"],
                "time_ms": (int(protocol["time"]) - experiment_start_time),  # in ms
            }
        )

    time_dict["protocol_times"] = time_list
    return time_dict


def convert_epoch_ms_to_datetime(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch / 1000.0)

    # create a time axis, using the protocol of the experiment. Values for pulse distance are


# in microseconds, so we need to convert them to milliseconds
def create_time_axis(label: str, iter: int, experiment_dict: dict) -> list:
    time_axis = []
    protocol_times = experiment_dict["time_dict"]["protocol_times"]
    protocolScript = getProtocolForLabel(label, experiment_dict["protocols"])
    print(f"protocolScript: {protocolScript}")

    if protocol_times[iter]["label"] != label:
        print(f"label mismatch: {protocol_times[iter]['label']} != {label}")
        return None
    print(f"protocol_times: {protocol_times}")

    pulses = parse_vlist(protocolScript["pulses"], experiment_dict["varray"])
    print(f"pulses: {pulses}")

    pulse_distance_us = parse_vlist(
        protocolScript["pulse_distance"], experiment_dict["varray"]
    )
    print(f"pulse_distance_us: {pulse_distance_us}")

    pulse_distance_ms = [x / 1000 for x in pulse_distance_us]
    print(f"pulse_distance_ms: {pulse_distance_ms}")

    trace_start_ms = protocol_times[iter]["time_ms"]
    print(f"trace_start_ms: {trace_start_ms}")

    # i for each unit of pulses, [500, 500, 500] for a 1500 pulse experiment
    # Each unit of pulses has a corresponding pulse_distance [1000, 1000, 1000] would be 1 ms pulse distance
    # We converted the us pulse distance to ms above, so we don't have to do it here
    last_time_point = trace_start_ms
    for i in range(len(pulses)):
        # j for each pulse in the range i of pulses, ex 0:500 for the first 500 pulses
        for j in range(pulses[i]):
            current_time_point = round(last_time_point + pulse_distance_ms[i], 4)
            time_axis.append(current_time_point)
            last_time_point = current_time_point

    print(f"time_axis: {time_axis}")

    if len(time_axis) != sum(pulses):
        print(f"length of time axis: {len(time_axis)} != sum of pulses: {sum(pulses)}")
        return None

    return time_axis


def deinterpolate_data(data: list, protocol: dict) -> dict:
    """
    Takes in a list of data and a protocol dict, and then deinterpolates it into a list of lists, where each list
    is a trace of data from a single light source. Then concatenates the lists into a single list
    of values, and returns it in a dict along with the lednums so you can tell which light source the data came from.
    """
    pulsed_lights = protocol.get("pulsed_lights", 0)
    pulse_phases = protocol.get("pulses", 0)

    # exit early if there are no pulsed lights or pulse phases
    if pulsed_lights == 0 or pulse_phases == 0:
        return None

    num_pulsed_leds = len(pulsed_lights[0])

    # exit early if there is only one light, no need to deinterpolate
    print(f"num_pulsed_leds: {num_pulsed_leds}")
    if num_pulsed_leds == 1:
        lednums = [pulsed_lights[0][0] for i in range(len(data))]
        return {"values": data, "lednums": lednums}

    total_pulses = num_pulsed_leds * sum(pulse_phases)

    print(f"num_pulsed_leds: {num_pulsed_leds}, total_pulses: {total_pulses}")

    # so now we can verify that our number is equal to the length of the data list
    if total_pulses != len(data):
        print(f"total_pulses: {total_pulses} != len(data): {len(data)}")
        return None

    # if the number of pulses is equal to the length of the data list, then we can
    # deinterpolate the data into a list of lists, where each list is a trace of data
    # from a single light source
    deinterpolated_data = [[] for i in range(num_pulsed_leds)]
    print(
        f"deinterpolated_data: {len(deinterpolated_data)}, num_pulsed_leds: {num_pulsed_leds}"
    )

    for i in range(total_pulses):
        # the order of the points is repeated, like: 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
        # so if i is 0, 2, 4, 6, 8, then we append that datapoint to the first list
        # if i is 1, 3, 5, 7, 9, then we append that datapoint to the second list
        if num_pulsed_leds == 1:
            deinterpolated_data[0].append(data[i])
            continue

        if i % 2 == 0:
            deinterpolated_data[0].append(data[i])
        else:
            deinterpolated_data[1].append(data[i])

    print(
        f"len(deinterpolated_data[0]): {len(deinterpolated_data[0])}, len(deinterpolated_data[1]): {len(deinterpolated_data[1])}"
    )

    # The length of each of those lists should be equal to the total_pulses / num_pulsed_leds
    if len(deinterpolated_data[0]) != total_pulses / num_pulsed_leds:
        print(
            f"len(deinterpolated_data[0]): {len(deinterpolated_data[0])} != total_pulses / num_pulsed_leds: {total_pulses / num_pulsed_leds}"
        )
        return None

    # concatenate the lists into a single list of values, with the first list first, and the second list second
    concatenated_data = deinterpolated_data[0] + deinterpolated_data[1]

    print(f"len(concatenated_data): {len(concatenated_data)}")

    concatenated_lednums = [pulsed_lights[0][0] for i in deinterpolated_data[0]] + [
        pulsed_lights[0][1] for i in deinterpolated_data[1]
    ]

    # return a dict of the concatenated data and the concatenated lednums
    return {"values": concatenated_data, "lednums": concatenated_lednums}


def convert_led_nums_to_wavelength(lednums: list) -> list:
    return [light_dict[i] for i in lednums]


def create_df(iter: int, data: dict, experiment_dict: dict) -> pd.DataFrame:
    """
    This function will take in a data dict, and then figure out if it is a single trace or a multi-trace
    experiment. It will then create a pandas dataframe from the data, and return it.

    The "data_raw" will need to be split into its individual parts, and then a time axis will be created
    that can be applied to both parts.

    There will be a shared time axis and value column, with each trace having a wavelength column that lists
    the measuring light used was.

    params:
    iter: int, the index of the protocol in the experiment_dic
    data: dict, the data from the experiment
    experiment_dict: dict, the dict containing the experiment data parameters, like the protocols, varray, etc.

    returns:
    pd.DataFrame, a dataframe of the data from the experiment like: labels, time_ms, wavelength, value, experiment_name, trace_num
    """
    print(f"creating dataframe for {data['label']}, trace {iter}")
    protocolScript = getProtocolForLabel(data["label"], experiment_dict["protocols"])

    data_raw = deinterpolate_data(data["data_raw"], protocolScript)

    if data_raw == None:
        print(f"deinterpolate_data returned None for {data['label']}, trace {iter}")
        return None

    values = data_raw["values"]

    wavelength = convert_led_nums_to_wavelength(data_raw["lednums"])

    labels = [data["label"] for j in range(len(values))]

    time_ms = create_time_axis(data["label"], iter, experiment_dict)

    trace_num = [i for i in range(len(data["data_raw"]))]

    print(f"creating dataframe for {data['label']}, trace {iter}")

    print(
        f"len(values): {len(values)}, len(labels): {len(labels)}, len(time_ms): {len(time_ms)}"
    )

    if len(time_ms) != len(values):
        time_ms = time_ms + time_ms

    print(len(time_ms), len(values))

    # adjust the time_ms to start at 0
    trace_start = min(time_ms)
    time_ms_trace = [t - trace_start for t in time_ms]

    data_dict = {
        "labels": labels,
        "time_ms": time_ms,
        "time_ms_trace": time_ms_trace,
        "wavelength": wavelength,
        "value": values,
        "experiment_name": experiment_dict["experiment_name"],
        "trace_num": trace_num,
    }

    return pd.DataFrame(data_dict)


def get_true_zero(df: pd.DataFrame) -> int:
    # get the mean of the values in which the labels fields includes "trueZero"
    # trueZero_noSP_noML_noActinic
    trueZero = df[p700_df["labels"].str.contains("trueZero")]["value"].mean()
    print(f"trueZero: {trueZero}")
    return trueZero


# The above code defines a function `calculate_fvfm_from_dataframe` that calculates the Fv/Fm ratio from a given DataFrame `df` and a true zero value. The function calculates the minimal fluorescence (Fo) and maximum fluorescence (Fm) values from specific rows in the DataFrame, subtracts the true zero value from these calculated values, and then computes the Fv/Fm ratio using the formula (Fm - Fo) / Fm. The function returns the calculated Fv/Fm ratio rounded to two decimal places.

# def calculate_fvfm_from_dataframe(df: pd.DataFrame, true_zero: int) -> float:
#     # Calculate Fv/Fm
#     # Fv = Fm - Fo
#     # Fm = max fluorescence
#     # Fo = minimal fluorescence
#     # Fv/Fm = (Fm - Fo) / Fm    # Fo is the minimal fluorescence measured in the dark-adapted state

#     # subtract the offset from true zero, and calculate the mean of the values

#     Fo = df.iloc[25:30]["value"].mean() - true_zero
#     Fm = df.iloc[40:50]["value"].mean() - true_zero

#     print(f"Fo: {Fo}, Fm: {Fm}")

#     return round((Fm - Fo) / Fm, 2)  # two decimal places is fine


def get_plant_species(filename: str) -> str:
    if "maize" in filename:
        return "maize"
    else:
        return "rice"


def get_genotype(plant_species: str, filename: str) -> str:
    plant_genotype = filename.split("\\")[-1].split(".json")[0]

    if "maize" in plant_genotype:
        plant_genotype = plant_genotype.split("maize")[1]
    else:
        plant_genotype = plant_genotype.split("_")[0]

    return plant_genotype


def get_plant_id(plant_species: str, filename: str) -> str:
    plant_id = filename.split("\\")[-1].split(".json")[0]

    # expect two underscores. If there are more or less, throw an error
    if plant_id.count("_") != 2:
        raise ValueError("plant_id does not have two underscores")
    
    # split the plant_id into parts, should be the second one
    parts = plant_id.split("_")
    plant_id = parts[1]

    return plant_id


def get_p_vals(
    data: pd.DataFrame,
    pm_mask_range: Tuple[int, int],
    p_mask_range: Tuple[int, int],
    po_mask_range: Tuple[int, int],
    x: str = "time_ms_trace",
    y: str = "dAbs",
) -> Tuple[float, float, float]:
    pm_mask = (data[x] >= pm_mask_range[0]) & (data[x] <= pm_mask_range[1])
    p_mask = (data[x] >= p_mask_range[0]) & (data[x] <= p_mask_range[1])
    po_mask = (data[x] >= po_mask_range[0]) & (data[x] <= po_mask_range[1])

    # Use the masks to filter the "ox_fraction" column and calculate the max and mean values
    Pmp = data.loc[pm_mask, y].max()

    P = data.loc[p_mask, y].dropna().mean()

    Po = data.loc[po_mask, y].mean()

    return (Pmp, P, Po)


def main():

    pm_mask_range = (495, 505)
    p_mask_range = (250, 495)
    po_mask_range = (1150, 1250)

    print(f"running main in {os.getcwd()}")
    files = glob.glob("data/*.json")
    print(f"files: {files}")

    for i, filename in enumerate(files):

        plant_species = get_plant_species(filename)
        
        genotype = get_genotype(plant_species, filename)
        plant_id = get_plant_id(plant_species, filename)

        print(f"\nFilename: {filename}")
        print(f"Plant Species: {plant_species}")
        print(f"Genotype: {genotype}")
        print(f"Plant ID: {plant_id}")

        with open(filename) as file:

            root = json.load(file)[0]

        print(f"Filename: {filename}")
        # whats the current status of this ovject? any keys?
        print(root.keys())

        time_dict = get_experiment_start_times_ms(filename)
        print(time_dict)

        # get the data from the file and parse it into a dict, or return "blank"
        time_dict = get_experiment_start_times_ms(filename)
        start = time_dict["experiment_start"]
        data_set = get_data_from_file(filename)
        protocol_dict = get_protocol_from_file(filename)
        experiment_name = get_experiment_name(filename)
        try:
            varray = parse_varray(protocol_dict["v_arrays"])
        except KeyError:
            print("no varray found")
            varray = None

        # create a dict to hold all the experiment data and parameters
        experiment_dict = {
            "data_set": data_set,
            "protocols": protocol_dict,
            "experiment_name": experiment_name,
            "experiment_date": experiment_name[:8],
            "experiment": experiment_name[9:],
            "varray": varray,
            "start_time": start,
            "time_dict": time_dict,
            "true_zero": -156,
        }
        print(experiment_dict["time_dict"])
        print(experiment_name)
        print(experiment_dict["protocols"])
        print(experiment_dict["experiment"])
        print(experiment_dict["experiment_date"])

        # ## Process the data and create the unified time series dataset

        # create an empty dataframe
        df = pd.DataFrame()
        for i, d in enumerate(data_set):
            print("i:", i)

            d_df = create_df(i, d, experiment_dict)
            if d_df is None:
                print(f"create_df returned None for {d['label']}, trace {i}")
                continue

            # check to see that the time_axis is a sequential list of numbers
            if (np.diff(d_df["time_ms"]) > 0).all():
                print("time axis is sequential")
            else:
                print("time axis is not sequential")

            print(d_df.head())

            # concatenate the dataframes
            df = pd.concat([df, pd.DataFrame(d_df)], ignore_index=True)

        df["time_s"] = round(df["time_ms"] / 1000, 6)

        print(df.head())
        print(df.tail())

        # adjust for true zero, at -156
        df["value"] = df["value"].apply(lambda x: x - experiment_dict["true_zero"])

        # remove all the "Preillum" labels, we don't need them
        df = df[~df["labels"].str.contains("Preillum")]

        # add the plant_species, plant_genotype, and plant_id to the dataframe
        df["plant_species"] = plant_species
        df["plant_genotype"] = genotype
        df["plant_id"] = plant_id

        # save the data to disk
        df.to_csv(f"output_data/{experiment_dict["experiment_name"]}.csv")

        # export the protocol to a json file
        with open(f"output_protocol/{experiment_dict["experiment_name"]}_protocol.json", "w") as file:
            json.dump(protocol_dict, file)

        data = pd.read_csv(f"output_data/{experiment_dict["experiment_name"]}.csv")

        print(f"data.columns= {data.columns}")

        # Set the aesthetic style of the plots
        sns.set_style("whitegrid")

        # Identify unique labels and wavelengths in the dataset
        unique_labels = data["labels"].unique()
        unique_wavelengths = data["wavelength"].unique()

        unique_labels, unique_wavelengths

        # Create a figure to hold the adjusted plots
        fig, axes = plt.subplots(len(unique_labels), 2, figsize=(12, 24), sharex=False)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        # Loop through each label and wavelength to create adjusted plots
        for i, label in enumerate(unique_labels):
            for j, wavelength in enumerate(unique_wavelengths):
                # Filter the data for the current label and wavelength
                filtered_data = data[
                    (data["labels"] == label) & (data["wavelength"] == wavelength)
                ]

                # Plot value over time with adjusted x-axis
                sns.lineplot(
                    ax=axes[i, j],
                    x="time_s",
                    y="value",
                    data=filtered_data,
                    linestyle="-",
                    linewidth=2,
                )

                # Set the title and labels
                axes[i, j].set_title(f"{label}, Wavelength: {wavelength}")
                axes[i, j].set_xlabel("Time (s)")
                axes[i, j].set_ylabel("Value")

                # Adjust the x-axis to show only the range with data
                if not filtered_data.empty:
                    min_time = filtered_data["time_s"].min()
                    max_time = filtered_data["time_s"].max()
                    axes[i, j].set_xlim(min_time, max_time)

        # Adjust layout to make room for titles and labels
        plt.tight_layout()

        # Show the adjusted plots
        # plt.show()

        # ## P700 data over time

        data_p700 = pd.read_csv(f"output_data/{experiment_name}.csv")
        true_zero = 0

        # Filter by wavelength
        data_fluor = data_p700[data_p700["wavelength"] == "530nm"]
        data_p700 = data_p700[data_p700["wavelength"] == "820nm"]

        # get the mean, max and min of the values
        p700_min, p700_max, p700_mean = (
            data_p700["value"].min(),
            data_p700["value"].max(),
            data_p700["value"].mean(),
        )

        # p700 min is the max of either the min value, or the mean * 0.99
        p700_min = min(p700_min - 100, p700_mean * 0.99)

        # p700 max is the min of either the max value, or the mean * 1.01
        p700_max = max(p700_max + 100, p700_mean * 1.01)

        print(f"p700_min: {p700_min}, p700_max: {p700_max}")

        # get unique trace labels
        sorted_labels = sorted(data_p700["labels"].unique())
        unique_labels = []

        # we want the reference first, then pmax, then the rest in order of the numver after "p700_"
        # reference is last, so we pop it and insert it at the beginning
        unique_labels.insert(0, sorted_labels.pop())

        # now we want to pop the pmax and insert it at the beginning, after the reference
        unique_labels.insert(1, sorted_labels.pop())

        # now the first two are in the right order, but the others are not.
        # we want to sort the rest of the labels by the number after "p700_"
        sorted_labels = sorted(sorted_labels, key=lambda x: int(x.split("_")[-1]))

        # now we want to insert the sorted labels into the unique_labels list, after the pmax
        for label in sorted_labels:
            unique_labels.append(label)

        print(unique_labels)

        # get fvfm from the fvfm trace, for the max value
        fvfm = calculate_fvfm(data_fluor[data_fluor["labels"] == "fvfm"], 100, 150)

        # get the pmax, and calculate the maximum value during the pulse
        pmax = data_p700[data_p700["labels"] == "p700_PMAX"]
        pmax = pmax.iloc[-150:140]["value"].mean()

        print(f"pmax: {pmax}")

        # plot the data
        plt.figure(figsize=(30, 10))

        fluor_data = {}

        for i, label in enumerate(unique_labels):
            ax = plt.subplot(1, len(unique_labels), i + 1)

            label_data = data_p700[data_p700["labels"] == label]
            label_data = label_data.iloc[10:]

            if "PMAX" in label:
                text_annotation = f"Fv/Fm: {fvfm}"
                fluor_data["fvfm"] = fvfm
            else:
                phi2 = calculate_fvfm(
                    data_fluor[data_fluor["labels"] == label], 100, 150
                )
                text_annotation = f"Phi2: {phi2}"
                fluor_data[label.split("_")[-1]] = phi2

            sns.lineplot(
                x="time_s", y="value", data=label_data, linestyle="-", linewidth=2
            )
            ax.set_title(f"P700 820nm {label}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value")
            ax.text(
                0.7,
                0.4,
                text_annotation,
                transform=ax.transAxes,
                fontsize=19,
                verticalalignment="top",
            )

            ax.set_ylim(p700_min, p700_max)

        plt.suptitle(f"{experiment_name} P700 820nm", fontsize=20)

        plt.tight_layout()
        plt.savefig(f"output_plots/{experiment_name}_820nm.png")

        # plt.show()
        # export the fluor_data dict  as a dataframe and save to csv
        # the dict should be a column marked "ue" with the labels as the index, and the values as the values
        fluor_df = pd.DataFrame(fluor_data, index=[0])

        # add the plant_species, plant_genotype, and plant_id to the dataframe
        fluor_df["plant_species"] = plant_species
        fluor_df["plant_genotype"] = genotype
        fluor_df["plant_id"] = plant_id

        fluor_df.to_csv(f"output_data/{experiment_name}_fluor.csv")
        print(fluor_df.transpose().head())

        # ### Calculation of the P700 values Pm, Pm', P, and P0
        # calculate dAbs for the traces

        # take the non-normalized data, and calculate the dAbs for each label
        data = pd.read_csv(f"output_data/{experiment_name}.csv")

        # remove all points  less than  10 time_ms_trace
        data = data[data["time_ms_trace"] > 10]

        # check the true zero to see if we're going crazy
        true_zero = data[data["labels"].str.contains("artifact")]["value"][0:100].mean()

        # Filter by wavelength
        data_fluor = data[data["wavelength"] == "530nm"]
        data_p700 = data[data["wavelength"] == "820nm"]

        # pulse start at 500ms and ends at 750ms
        pulse_start = 500
        pulse_end = 750

        # Calculate dAbs for each label and save it in the DataFrame
        for label in unique_labels:
            # Filter data for the current label
            label_data = data_p700[data_p700["labels"] == label].copy()

            # Calculate baseline for the current label
            baseline_value = calculate_baseline_for_label(label_data, -20, -5)

            # Calculate dAbs for each row in the label_data
            label_data["dAbs"] = calc_delta_absorbance(
                label_data["value"].values, baseline_value
            )

            # Update the main DataFrame (data_p700) with the calculated dAbs values for the current label
            data_p700.loc[data_p700["labels"] == label, "dAbs"] = label_data["dAbs"]

        # add the plant_species, plant_genotype, and plant_id to the dataframe

        data_p700["plant_species"] = plant_species
        data_p700["plant_genotype"] = genotype
        data_p700["plant_id"] = plant_id

        data_p700.to_csv(f"output_data/{experiment_name}_dAbs.csv")

        # plot dAbs for 250
        plt.figure(figsize=(30, 10))

        # get the mean, max and min of the values
        p700_min, p700_max = (
            data_p700["dAbs"].min() - 0.005,
            data_p700["dAbs"].max() + 0.005,
        )

        for i, label in enumerate(unique_labels):

            ax = plt.subplot(1, len(data_p700["labels"].unique()), i + 1)
            label_data = (
                data_p700[data_p700["labels"] == label].copy().reset_index(drop=True)
            )

            print(len(label_data))

            # # set a vertical line
            ax.axvline(pulse_start, color="red", linestyle="--")
            ax.axvline(pulse_end, color="red", linestyle="--")

            sns.lineplot(
                x="time_ms_trace", y="dAbs", data=label_data, linestyle="-", linewidth=2
            )
            ax.set_title(f"{label}", fontsize=16)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Abs")

            ax.set_ylim(p700_min, p700_max)

        plt.suptitle(f"{experiment_name} 820nm dAbs", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"output_plots/{experiment_name}_820nm_dAbs.png")
        # plt.show()

        data_p700 = pd.read_csv(f"output_data/{experiment_name}_dAbs.csv")

        # normalize the data using the values from the pmax trace
        pmax_df = data_p700[data_p700["labels"].str.contains("p700_PMAX")]

        # calculate p values for pmax
        Pm, P, Po = get_p_vals(pmax_df, pm_mask_range, p_mask_range, po_mask_range)
        print(f"Pm: {Pm}, P: {P}, Po: {Po}")

        data_p700["ox_fraction"] = (data_p700["dAbs"] - Po) / (Pm - Po)

        # add the plant_species, plant_genotype, and plant_id to the dataframe
        data_p700["plant_species"] = plant_species
        data_p700["plant_genotype"] = genotype
        data_p700["plant_id"] = plant_id

        data_p700.to_csv(f"output_data/{experiment_name}_dAbs_normalized.csv")

        # take the normalized data and plot it, with the P values shown as horizontal lines
        data_p700 = pd.read_csv(f"output_data/{experiment_name}_dAbs_normalized.csv")

        # only if the wavelength is 820nm
        data_p700 = data_p700[data_p700["wavelength"] == "820nm"]

        # param locations in plot
        param_x = 900

        # list to save the P values
        p_vals = []

        # normalized data
        plt.figure(figsize=(30, 10))
        plt.suptitle(f"{experiment_name} P700 Oxidation", fontsize=20)

        # remove "reference" from the unique_labels
        unique_labels = [label for label in unique_labels if "reference" not in label]

        for i, label in enumerate(unique_labels):

            print(f"label: {label}")

            if "PMAX" in label:
                title_label = f"P700 @ {label}"
            else:
                light_intensity = label.split("_")[-1]
                title_label = f"P700 @ {light_intensity}ue"

            ax = plt.subplot(1, len(unique_labels), i + 1)
            label_data = data_p700[data_p700["labels"] == label]
            label_data = label_data.iloc[10:]
            label_data = label_data.reset_index(drop=True)

            if "PMAX" not in label:
                # calculate the Pmp, P, Po for all of the non-PMAX traces

                Pmp, P, Po = get_p_vals(label_data, pm_mask_range, p_mask_range, po_mask_range, x="time_ms_trace", y="ox_fraction")
                Pm = 1.0

                print(f"Pm: {Pm}, Pmp: {Pmp}, P: {P}, Po: {Po}")
                p_vals.append(
                    {
                        "ue": light_intensity,
                        "Pm": Pm,
                        "Pmp": Pmp,
                        "P": P,
                        "Po": Po,
                        "Y(I)": Pmp - P,
                        "Y(ND)": P - Po,
                        "Y(NA)": Pm - Pmp,
                    }
                )

            # plot the normalized data as lines, with points as well
            sns.lineplot(
                x="time_ms_trace",
                y="ox_fraction",
                data=label_data,
                linestyle="-",
                linewidth=2,
                markersize=5,
                marker="o",
            )

            # if the label is not PMAX, then plot these:
            if "PMAX" not in label:
                ax.hlines(Pm, 0, 1300, color="red", linestyle="--")  # Pm
                ax.hlines(Pmp, 0, 1300, color="green", linestyle="--")  # Pm'
                ax.hlines(P, 0, 1300, color="blue", linestyle="--")  # P
                ax.hlines(Po, 0, 1300, color="black", linestyle="--")  # Po

                # put a P next to the P line, a P' next to the P' line, etc
                ax.text(0, Pm, "Pm", fontsize=14)
                ax.text(0, Pmp, "Pmp", fontsize=14)
                ax.text(0, P, "P", fontsize=14)
                ax.text(0, Po, "Po", fontsize=14)
            else:
                ax.set_ylabel("Fraction of P700 oxidized", fontsize=14)
                ax.hlines(
                    1, 0, 1300, color="red", linestyle="--"
                )  # Everything is oxidized

            # now we need to add text annotations for the Y() values onto the plot, for each trace
            if "PMAX" not in label:
                ax.text(
                    param_x,
                    Pm - 0.05,
                    f"Y(NA): {round(1 - Pmp, 2)}",
                    fontsize=14,
                    horizontalalignment="left",
                )
                ax.text(
                    param_x,
                    Pmp - 0.05,
                    f"Y(I): {round(Pmp - P, 2)}",
                    fontsize=14,
                    horizontalalignment="left",
                )
                ax.text(
                    param_x,
                    P - 0.05,
                    f"Y(ND): {round(P, 2)}",
                    fontsize=14,
                    horizontalalignment="left",
                )
            ax.set_ylim(
                -0.1, 1.1
            )  # set the y-axis limits, so we can see how the data changes from the normalized value
            ax.set_title(f"{title_label}", fontsize=14)
            ax.set_xlabel("Time (ms)")

        plt.tight_layout()
        plt.savefig(f"output_plots/{experiment_name}_p700_oxidation_normalized.png")
        # plt.show()
        print(p_vals)

        # create a dataframe of the p_vals, save it
        p_vals_df = pd.DataFrame(p_vals)

        # add the plant_species, plant_genotype, and plant_id to the dataframe
        p_vals_df["plant_species"] = plant_species
        p_vals_df["plant_genotype"] = genotype
        p_vals_df["plant_id"] = plant_id

        p_vals_df.to_csv(f"output_data/{experiment_name}_p_vals.csv")

        # take the normalized data and plot it, with the P values shown as horizontal lines
        data_p700 = pd.read_csv(f"output_data/{experiment_name}_dAbs_normalized.csv")

        # only if the wavelength is 820nm
        data_p700 = data_p700[data_p700["wavelength"] == "820nm"]

        # param locations in plot
        param_x = 900

        # list to save the P values
        p_vals = []

        # normalized data
        plt.figure(figsize=(20, 10))
        plt.suptitle(f"{experiment_name} P700 Oxidation", fontsize=20)

        # remove "reference" from the unique_labels
        unique_labels = [label for label in unique_labels if "reference" not in label]

        for i, label in enumerate(unique_labels):

            print(f"label: {label}")

            if "PMAX" in label:
                title_label = f"P700 @ {label}"
            else:
                light_intensity = label.split("_")[-1]
                title_label = f"P700 @ {light_intensity}ue"

            ax = plt.subplot(1, len(unique_labels), i + 1)
            label_data = data_p700[data_p700["labels"] == label]
            label_data = label_data.iloc[10:]
            label_data = label_data.reset_index(drop=True)

            if "PMAX" not in label:
                # calculate the Pmp, P, Po for all of the non-PMAX traces
                Pmp, P, Po = get_p_vals(label_data, pm_mask_range, p_mask_range, po_mask_range, x="time_ms_trace", y="ox_fraction")
                Pm = 1.0

                print(f"Pm: {Pm}, Pmp: {Pmp}, P: {P}, Po: {Po}")
                p_vals.append(
                    {
                        "ue": light_intensity,
                        "Pm": Pm,
                        "Pmp": Pmp,
                        "P": P,
                        "Po": Po,
                        "Y(I)": Pmp - P,
                        "Y(ND)": P - Po,
                        "Y(NA)": Pm - Pmp,
                    }
                )

            # plot the normalized data as lines, with points as well
            sns.lineplot(
                x="time_ms_trace",
                y="ox_fraction",
                data=label_data,
                linestyle="-",
                linewidth=2,
                markersize=5,
                marker="o",
            )

            ax.set_xlim(pm_mask_range[0] - 50, pm_mask_range[1] + 50)

            # if the label is not PMAX, then plot these:
            if "PMAX" not in label:
                ax.hlines(Pm, 0, 1300, color="red", linestyle="--")  # Pm
                ax.hlines(Pmp, 0, 1300, color="green", linestyle="--")  # Pm'
                ax.hlines(P, 0, 1300, color="blue", linestyle="--")  # P
                ax.hlines(Po, 0, 1300, color="black", linestyle="--")  # Po

                # put a P next to the P line, a P' next to the P' line, etc
                ax.text(pm_mask_range[0] - 50, Pm, "Pm", fontsize=14)
                ax.text(pm_mask_range[0] - 50, Pmp, "Pmp", fontsize=14)
                ax.text(pm_mask_range[0] - 50, P, "P", fontsize=14)
                ax.text(pm_mask_range[0] - 50, Po, "Po", fontsize=14)
            else:
                ax.set_ylabel("Fraction of P700 oxidized", fontsize=14)
                ax.hlines(
                    1, 0, 1300, color="red", linestyle="--"
                )  # Everything is oxidized

            # now we need to add text annotations for the Y() values onto the plot, for each trace
            if "PMAX" not in label:
                ax.text(
                    pm_mask_range[1] + 10,
                    Pm - 0.05,
                    f"Y(NA): {round(1 - Pmp, 2)}",
                    fontsize=14,
                    horizontalalignment="left",
                )
                ax.text(
                    pm_mask_range[1] + 10,
                    Pmp - 0.05,
                    f"Y(I): {round(Pmp - P, 2)}",
                    fontsize=14,
                    horizontalalignment="left",
                )
                ax.text(
                    pm_mask_range[1] + 10,
                    P - 0.05,
                    f"Y(ND): {round(P, 2)}",
                    fontsize=14,
                    horizontalalignment="left",
                )
            ax.set_ylim(
                -0.1, 1.1
            )  # set the y-axis limits, so we can see how the data changes from the normalized value
            ax.set_title(f"{title_label}", fontsize=14)
            ax.set_xlabel("Time (ms)")

        plt.tight_layout()
        plt.savefig(
            f"output_plots/{experiment_name}_p700_oxidation_normalized_zoom.png"
        )
        # plt.show()
        print(p_vals)

        # create a dataframe of the p_vals, save it
        p_vals_df = pd.DataFrame(p_vals)

        # add the plant_species, plant_genotype, and plant_id to the dataframe
        p_vals_df["plant_species"] = plant_species
        p_vals_df["plant_genotype"] = genotype
        p_vals_df["plant_id"] = plant_id

        p_vals_df.to_csv(f"output_data/{experiment_name}_p_vals.csv")

        # ### TODO:
        #
        # Calculate p' whichever is higher, last few datapoints of p or a mean of the first few datapoints during the pulse

        # ### adjust the pmax calculation
        #
        # Either take the mean of the first few points of the pulse, or the mean of the last few points of the prepulse, whichever is higher.

        drop_columns = ["Unnamed: 0", "plant_species", "plant_genotype", "plant_id"]

        # load the p_vals dataframe
        p_vals_df = pd.read_csv(f"output_data/{experiment_name}_p_vals.csv")
        p_vals_df = p_vals_df.drop(columns=drop_columns)

        # load the fluor data as well
        fluor_df = pd.read_csv(f"output_data/{experiment_name}_fluor.csv")
        fluor_df = fluor_df.drop(columns=drop_columns).transpose()

        # drop all rows that have label including "reference"
        fluor_df = fluor_df[~fluor_df.index.str.contains("reference")]

        # index to column
        fluor_df = fluor_df.reset_index()

        # rename the columns
        fluor_df.columns = ["ue", "value"]

        # extract the fvfm value for later use
        fvfm = fluor_df[fluor_df["ue"] == "fvfm"]["value"].values[0]

        # slice teh dataframe to get all ue that are not fvfm
        fluor_df = fluor_df[fluor_df["ue"] != "fvfm"]

        print(fluor_df.head())

        # convert the ue column to int64
        fluor_df["ue"] = fluor_df["ue"].astype("int64").copy()

        # value is now the Phi2 value
        p_vals_df["PhiII"] = fluor_df["value"].values
        p_vals_df["FvFm"] = fvfm
        p_vals_df.head()

        # Now plot Phi2 vs Y(I)
        plt.figure(figsize=(5, 5))
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        # plot the Phi2 vs Y(I)
        sns.scatterplot(x="PhiII", y="Y(I)", data=p_vals_df, s=100, color="blue")

        # add a title and labels
        plt.suptitle(f"{experiment_name}", fontsize=18)
        plt.title(f"Phi2 vs Y(I)", fontsize=18)

        # add labels
        plt.xlabel("Phi2", fontsize=20)
        plt.ylabel("Y(I)", fontsize=20)

        plt.savefig(f"output_plots/{experiment_name}_Phi2_vs_YI.png")
        # plt.show()        p_vals_df.head()

        # params for the p700 traces
        params = {}

        # calc pmax first
        pmax_df = data_p700[data_p700["labels"] == "p700_PMAX"]
        params["Pm"] = pmax_df["dAbs"].iloc[-150:-140].mean()
        params["Po"] = pmax_df["dAbs"].iloc[-20:].mean()

        for label in data_p700["labels"].unique():

            if "PMAX" not in label and "refer" not in label:
                print(f"label: {label}")
                df = data_p700[data_p700["labels"] == label]

                params[label] = {
                    "phi2": calculate_fvfm(
                        data_fluor[data_fluor["labels"] == label], 100, 150
                    ),
                    "P": df["dAbs"].iloc[-170:-150].mean(),
                    "Pmp": df["dAbs"].iloc[-150:-140].mean(),
                    "Po": df["dAbs"].iloc[-20:].mean(),
                }

        print(params)

        def remove_y_axis(ax):
            # Remove tick marks and labels from the y-axis
            ax.set_ylabel("")
            ax.set_yticklabels([])


        data_csv_fluor_2 = pd.read_csv(f"output_data/{experiment_name}.csv")

        # Filter by wavelength
        data_green = data_csv_fluor_2[data_csv_fluor_2["wavelength"] == "530nm"]

        # sort for the order of the traces
        sorted_labels = sorted(data_green["labels"].unique())

        sorted_labels = [
            label
            for label in sorted_labels
            if ("fvfm" in label or "p700" in label or "fluor" in label)
            and ("PMAX" not in label)
        ]

        unique_labels = []
        light_labels = []

        for label in sorted_labels:
            if "p700" in label:
                light_labels.append(label)
            else:
                unique_labels.append(label)

        sorted_labels = sorted(light_labels, key=lambda x: int(x.split("_")[-1]))

        unique_labels.extend(sorted_labels)

        print(unique_labels)

        unique_labels_green = unique_labels
        plt.figure(figsize=(30, 10))

        # get the ylim values for the fvfm trace, by finding the max value for the "value" column where the label == "fvfm"
        label_data = data_green[data_green["labels"] == "fvfm"]
        ylim = [0, np.max(label_data["value"]) * 1.1]

        # Plot green data
        for i, label in enumerate(unique_labels_green):

            ax = plt.subplot(1, len(unique_labels_green), i + 1)
            label_data = data_green[data_green["labels"] == label]
            fvfm = calculate_fvfm(label_data, 100, 150)
            sns.lineplot(
                x="time_s", y="value", data=label_data, linestyle="-", linewidth=2
            )
            ax.set_title(f"Fluor {label} (Fv/Fm: {fvfm})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value")

            # else:
            #     remove_y_axis(ax)

            ax.set_ylim(ylim[0], ylim[1])

            # Add Fv/Fm text annotation within the plot
            ax.text(
                0.6,
                0.9,
                f"Fv/Fm: {fvfm}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
            )

        plt.suptitle(f"{experiment_name} Fluor", fontsize=20)

        plt.tight_layout()
        plt.savefig(f"output_plots/{experiment_name}_fluor.png")
# plt.show()

# # c


if __name__ == "__main__":
    main()