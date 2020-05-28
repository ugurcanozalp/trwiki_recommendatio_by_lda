import yaml
import sys
import os
import glob
import pandas as pd


def configloader():
    # Read Config YAML file.
    with open("config.yml", 'r') as stream:
        cfg = yaml.safe_load(stream)

    return cfg

def folderloader(datafolder):
    # import data of overall corpus csv files.
    data_query = os.path.join(datafolder,'*.csv')
    files = glob.glob(data_query)
    # loop over csv files and stack them.
    data = pd.Series()
    for filename in files:
        data_tmp = pd.read_csv(filename)
        if 'text' not in data_tmp.columns:
            print(f"WARNING: {filename} has not column called 'text'..!")
            continue
        data = pd.concat([data,data_tmp['text']])
    # Reset indexes
    data.reset_index(drop=True,inplace=True)
    return data

def fileloader(filename):
    data = pd.read_csv(filename)
    return data