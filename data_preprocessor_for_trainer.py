"""
DATA PREPROCESSOR FOR TRAINER

Developed By: Thomas Vaughn
Version: 1.1.4
Last Update Date: 7/10/2025

This script is designed to be imported as a module from the
Product Classifier Trainer script.

The prepare_dataset function prepares a dataset for training a
Product Classifier model.  It splits a CSV file into separate
train, validation, and test splits using the Hugging Face Datasets
library.  Then it performs several specification translation and
generalization activities which support the Product Classification
task including:

    * Converting Category IDs to Classifier Labels
    * Converting pack sizes to generalized terms
    * Converting storage temperatures to generalized storage terms

This file contains the following function:

    * prepare_dataset - Prepares a dataset for training a Product Classification model

"""


from datasets import load_dataset
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from data_cleaner import clean_data, clean_first_pass, clean_none, prune_data_final
import math

pd.set_option('max_colwidth', 175)

label_mapping_fpath = './source_data/category_to_label_mapping.csv'
raw_train_fpath = 'train_raw_dataset.csv'
raw_validation_fpath = 'validation_raw_dataset.csv'
raw_test_fpath = 'test_raw_dataset.csv'

#####
# Adjust as desired
#
_train_percent = 0.7


def prepare_dataset(source_fpath):
    """

    :param source_fpath:
    :return:
    """

    raw_data = pd.read_csv(source_fpath, encoding='utf-8', dtype=str)

    print("\nPerforming first pass data cleansing...\n")

    first_pass_cleaned_data = clean_first_pass(raw_data)

    # Strip the file extension from the filepath
    len_stripped_fpath = len(source_fpath) - 4
    stripped_fpath = source_fpath[:len_stripped_fpath]

    # Build the prepped data filepath
    first_pass_fpath = stripped_fpath + "_first_pass_cleansed.csv"

    first_pass_cleaned_data.to_csv(first_pass_fpath, sep=',', encoding='utf-8', index=False)

    original = load_dataset(
        'csv',
        data_files=first_pass_fpath,
        delimiter=',',
        column_names=['product_number', 'description', 'manufacturer', 'brand', 'coo',
                      'pack_size', 'temp_min', 'temp_min_uom', 'temp_max', 'temp_max_uom'],
        skiprows=1
    )

    #####
    # use train_test_split to create train, validation, and test splits
    #

    print("\nSplitting data into train, validation, and test datasets...\n")

    # shuffle the original dataset
    original = original.shuffle(seed=1)

    # split the shuffled dataset into train/test based on the value of the _train_percent variable
    dataset = original['train'].train_test_split(train_size=_train_percent)

    # create a new temporary dataset to split the test split into 2 splits
    # this will result in a train and test split here, too
    temporary_test_validation = dataset['test'].train_test_split(train_size=0.5)

    # pop the test split from the main dataset since we will be using the splits below
    dataset.pop('test')

    # create a validation split based on the temporary_test_validation 'train' split
    # this will result in a train, test, and validation split (copy of train)
    temporary_test_validation['validation'] = temporary_test_validation['train']

    # pop the 'train' split from the temporary dataset
    temporary_test_validation.pop('train')

    # update the main dataset to include the new test and validation splits
    dataset.update(temporary_test_validation)

    #####
    # Converting dataset to pandas format to prepare features for training
    #

    dataset.set_format('pandas')

    raw_train_df = dataset['train'][:]
    raw_validation_df = dataset['validation'][:]
    raw_test_df = dataset['test'][:]

    raw_train_df.to_csv(raw_train_fpath, sep=',', encoding='utf-8', index=False)
    raw_validation_df.to_csv(raw_validation_fpath, sep=',', encoding='utf-8', index=False)
    raw_test_df.to_csv(raw_test_fpath, sep=',', encoding='utf-8', index=False)

    print("\nPerforming final pass data cleaning...\n")

    train_df = clean_data(raw_train_df)
    validation_df = clean_data(raw_validation_df)
    test_df = clean_data(raw_test_df)

    #####
    # Converting Category IDs to Training Labels
    #

    print("\nConverting Categories to training labels...\n")
    time.sleep(.3)

    label_df = pd.read_csv(label_mapping_fpath)
    label_map = {}

    # Populating the category map
    for i in range(label_df.shape[0]):
        category = label_df.loc[i, 'category']
        label = label_df.loc[i, 'label']
        label_map[category] = label

    # Converting Category IDs for Training Dataset
    for i in tqdm(range(train_df.shape[0])):
        category = train_df.loc[i, 'category']
        try:
            label = label_map[category]
        except:
            continue
        train_df.at[i, 'category'] = label

    # Converting Category IDs for Validation Dataset
    for i in tqdm(range(validation_df.shape[0])):
        category = validation_df.loc[i, 'category']
        try:
            label = label_map[category]
        except:
            continue
        validation_df.at[i, 'category'] = label

    # Converting Category IDs for Test Dataset
    for i in tqdm(range(test_df.shape[0])):
        category = test_df.loc[i, 'category']
        try:
            label = label_map[category]
        except:
            continue
        test_df.at[i, 'category'] = label

    #####
    # Converting pack sizes to general terms more easily managed by the model
    #

    print("\nConverting pack sizes to general terms...\n")
    time.sleep(.3)

    # Converting pack sizes to generalized terms for train dataset
    for i in tqdm(range(train_df.shape[0])):
        orig_pack = str(train_df.loc[i, 'pack_size'])
        try:
            # Check to see if uom is can
            if "can" in orig_pack:
                train_df.at[i, 'pack_size'] = "Can"
                continue

            has_two_levels = False
            has_three_levels = False

            # Parse the pack size
            slash_split_pack = orig_pack.split('/')  # parses by '/' and returns a list of the substrings

            if len(slash_split_pack) >= 3:
                has_three_levels = True
            elif len(slash_split_pack) == 2:
                has_two_levels = True

            lowest_level_pack = slash_split_pack[len(slash_split_pack) - 1]  # capture the last substring in the list
            whitespace_split_pack = lowest_level_pack.split()  # parses by whitespace and returns a list
            lowest_level_pack_value = whitespace_split_pack[0]  # index 0 s/b the value
            lowest_level_pack_uom = whitespace_split_pack[1]  # index 1 s/b the uom

            # Check for a dash (often used for ranges)
            dash_split = lowest_level_pack_value.split('-')
            if len(dash_split) > 1:
                lowest_level_pack_value = dash_split[len(dash_split) - 1]  # If there is more than one value, use the last one

            # Convert the lowest level pack size value to a float for comparisons below
            lowest_level_pack_value = float(lowest_level_pack_value)

            # Run logic below to convert pack size to general terms
            try:
                if lowest_level_pack_uom == "lb" or lowest_level_pack_uom == "lbr" or \
                        lowest_level_pack_uom == "lba" or lowest_level_pack_uom == "pound":
                    if lowest_level_pack_value <= 1.0:
                        train_df.at[i, 'pack_size'] = "lessequalto1lb"
                    elif lowest_level_pack_value <= 5.0:
                        train_df.at[i, 'pack_size'] = "lessequalto5lb"
                    elif lowest_level_pack_value <= 10.0:
                        train_df.at[i, 'pack_size'] = "lessequalto10lb"
                    elif lowest_level_pack_value <= 20.0:
                        train_df.at[i, 'pack_size'] = "lessequalto20lb"
                    elif lowest_level_pack_value <= 100.0:
                        train_df.at[i, 'pack_size'] = "lessequalto100lb"
                    elif lowest_level_pack_value > 100.0:
                        train_df.at[i, 'pack_size'] = "greaterthan100lb"
                elif lowest_level_pack_uom == "oz" or lowest_level_pack_uom == "oza" or \
                     lowest_level_pack_uom == "onz" or lowest_level_pack_uom == "ounce" or \
                     lowest_level_pack_uom == "z" or lowest_level_pack_uom == "za":
                    if lowest_level_pack_value <= 1.0:
                        train_df.at[i, 'pack_size'] = "lessequalto1oz"
                    elif lowest_level_pack_value <= 2.0:
                        train_df.at[i, 'pack_size'] = "lessequalto2oz"
                    elif lowest_level_pack_value <= 3.0:
                        train_df.at[i, 'pack_size'] = "lessequalto3oz"
                    elif lowest_level_pack_value <= 4.0:
                        train_df.at[i, 'pack_size'] = "lessequalto4oz"
                    elif lowest_level_pack_value <= 5.0:
                        train_df.at[i, 'pack_size'] = "lessequalto5oz"
                    elif lowest_level_pack_value <= 6.0:
                        train_df.at[i, 'pack_size'] = "lessequalto6oz"
                    elif lowest_level_pack_value <= 8.0:
                        train_df.at[i, 'pack_size'] = "lessequalto8oz"
                    elif lowest_level_pack_value <= 12.0:
                        train_df.at[i, 'pack_size'] = "lessequalto12oz"
                    elif lowest_level_pack_value <= 16.0:
                        train_df.at[i, 'pack_size'] = "lessequalto16oz"
                    elif lowest_level_pack_value <= 20.0:
                        train_df.at[i, 'pack_size'] = "lessequalto20oz"
                    elif lowest_level_pack_value <= 24.0:
                        train_df.at[i, 'pack_size'] = "lessequalto24oz"
                    elif lowest_level_pack_value <= 32.0:
                        train_df.at[i, 'pack_size'] = "lessequalto32oz"
                    elif lowest_level_pack_value > 32.0:
                        train_df.at[i, 'pack_size'] = "lessequalto100oz"
                    elif lowest_level_pack_value <= 64.0:
                        train_df.at[i, 'pack_size'] = "lessequalto64oz"
                    elif lowest_level_pack_value <= 100.0:
                        train_df.at[i, 'pack_size'] = "lessequalto100oz"
                    elif lowest_level_pack_value > 100.0:
                        train_df.at[i, 'pack_size'] = "greaterthan100oz"
                elif lowest_level_pack_uom == "ea" or lowest_level_pack_uom == "ct" or \
                        lowest_level_pack_uom == "count" or lowest_level_pack_uom == "each":
                    if lowest_level_pack_value <= 1.0:
                        train_df.at[i, 'pack_size'] = "oneeach"
                    elif lowest_level_pack_value <= 5.0:
                        train_df.at[i, 'pack_size'] = "lessthan5ea"
                    elif lowest_level_pack_value <= 10.0:
                        train_df.at[i, 'pack_size'] = "lessthan10ea"
                    elif lowest_level_pack_value <= 15.0:
                        train_df.at[i, 'pack_size'] = "lessthan15ea"
                    elif lowest_level_pack_value <= 20.0:
                        train_df.at[i, 'pack_size'] = "lessthan20ea"
                    elif lowest_level_pack_value <= 25.0:
                        train_df.at[i, 'pack_size'] = "lessthan25ea"
                    elif lowest_level_pack_value <= 30.0:
                        train_df.at[i, 'pack_size'] = "lessthan30ea"
                    elif lowest_level_pack_value <= 40.0:
                        train_df.at[i, 'pack_size'] = "lessthan40ea"
                    elif lowest_level_pack_value <= 50.0:
                        train_df.at[i, 'pack_size'] = "lessthan50ea"
                    elif lowest_level_pack_value <= 75.0:
                        train_df.at[i, 'pack_size'] = "lessthan75ea"
                    elif lowest_level_pack_value <= 100.0:
                        train_df.at[i, 'pack_size'] = "lessthan100ea"
                    elif lowest_level_pack_value <= 150.0:
                        train_df.at[i, 'pack_size'] = "lessthan150ea"
                    elif lowest_level_pack_value <= 200.0:
                        train_df.at[i, 'pack_size'] = "lessthan200ea"
                    elif lowest_level_pack_value <= 250.0:
                        train_df.at[i, 'pack_size'] = "lessthan250ea"
                    elif lowest_level_pack_value <= 500.0:
                        train_df.at[i, 'pack_size'] = "lessthan500ea"
                    elif lowest_level_pack_value <= 1000.0:
                        train_df.at[i, 'pack_size'] = "lessthan1000ea"
                    elif lowest_level_pack_value > 1000.0:
                        train_df.at[i, 'pack_size'] = "greaterthan1000ea"
                elif lowest_level_pack_uom == "dz" or lowest_level_pack_uom == "doz" or \
                        lowest_level_pack_uom == "dzn" or lowest_level_pack_uom == "dozen":
                    if lowest_level_pack_value <= 1.0:
                        train_df.at[i, 'pack_size'] = "TinyDozen"
                    elif lowest_level_pack_value <= 3.0:
                        train_df.at[i, 'pack_size'] = "LoDozen"
                    elif lowest_level_pack_value <= 6.0:
                        train_df.at[i, 'pack_size'] = "LoMedDozen"
                    elif lowest_level_pack_value <= 20.0:
                        train_df.at[i, 'pack_size'] = "MedDozen"
                    elif lowest_level_pack_value > 20.0:
                        train_df.at[i, 'pack_size'] = "HiDozen"
                elif lowest_level_pack_uom == "ga" or lowest_level_pack_uom == "gll" or \
                        lowest_level_pack_uom == "gal" or lowest_level_pack_uom == "gallon":
                    if lowest_level_pack_value <= 1.0:
                        train_df.at[i, 'pack_size'] = "TinyGallon"
                    elif lowest_level_pack_value <= 3.0:
                        train_df.at[i, 'pack_size'] = "LoGallon"
                    elif lowest_level_pack_value <= 5.0:
                        train_df.at[i, 'pack_size'] = "MedGallon"
                    elif lowest_level_pack_value > 5.0:
                        train_df.at[i, 'pack_size'] = "HiGallon"
                elif lowest_level_pack_uom == "rl" or lowest_level_pack_uom == "roll":
                    if lowest_level_pack_value <= 10.0:
                        train_df.at[i, 'pack_size'] = "LoRoll"
                    elif lowest_level_pack_value <= 50.0:
                        train_df.at[i, 'pack_size'] = "MedRoll"
                    elif lowest_level_pack_value > 50.0:
                        train_df.at[i, 'pack_size'] = "HiRoll"
                elif lowest_level_pack_uom == "cn" or lowest_level_pack_uom == "can":
                    train_df.at[i, 'pack_size'] = "Can"
                elif lowest_level_pack_uom == "lt" or lowest_level_pack_uom == "ltr" or \
                        lowest_level_pack_uom == "liter" or lowest_level_pack_uom == "litre":
                    if lowest_level_pack_value <= 1.0:
                        train_df.at[i, 'pack_size'] = "TinyLiter"
                    elif lowest_level_pack_value <= 4.0:
                        train_df.at[i, 'pack_size'] = "LoLiter"
                    elif lowest_level_pack_value <= 12.0:
                        train_df.at[i, 'pack_size'] = "MedLiter"
                    elif lowest_level_pack_value > 12.0:
                        train_df.at[i, 'pack_size'] = "HiLiter"
                elif lowest_level_pack_uom == "kg" or lowest_level_pack_uom == "kgr" or \
                     lowest_level_pack_uom == "kgm" or lowest_level_pack_uom == "kilo" or \
                     lowest_level_pack_uom == "kilogram":
                    if lowest_level_pack_value <= 3.0:
                        train_df.at[i, 'pack_size'] = "LoKilogram"
                    elif lowest_level_pack_value <= 12.0:
                        train_df.at[i, 'pack_size'] = "MedKilogram"
                    elif lowest_level_pack_value > 12.0:
                        train_df.at[i, 'pack_size'] = "HiKilogram"
                elif lowest_level_pack_uom == "qt" or lowest_level_pack_uom == "qtl" or \
                        lowest_level_pack_uom == "qtd" or lowest_level_pack_uom == "quart":
                    if lowest_level_pack_value <= 4.0:
                        train_df.at[i, 'pack_size'] = "LoQuart"
                    elif lowest_level_pack_value <= 8.0:
                        train_df.at[i, 'pack_size'] = "MedQuart"
                    elif lowest_level_pack_value > 8.0:
                        train_df.at[i, 'pack_size'] = "HiQuart"
                elif lowest_level_pack_uom == "pt" or lowest_level_pack_uom == "ptl" or \
                        lowest_level_pack_uom == "pti" or lowest_level_pack_uom == "pint":
                    if lowest_level_pack_value <= 1.0:
                        train_df.at[i, 'pack_size'] = "LoPint"
                    elif lowest_level_pack_value <= 4.0:
                        train_df.at[i, 'pack_size'] = "MedPint"
                    elif lowest_level_pack_value > 4.0:
                        train_df.at[i, 'pack_size'] = "HiPint"
                elif lowest_level_pack_uom == "kt" or lowest_level_pack_uom == "kit":
                    train_df.at[i, 'pack_size'] = "Kit"
                elif lowest_level_pack_uom == "ml" or lowest_level_pack_uom == "mlt" or \
                        lowest_level_pack_uom == "milliliter" or lowest_level_pack_uom == "millilitre":
                    if lowest_level_pack_value <= 25.0:
                        train_df.at[i, 'pack_size'] = "TinyMilliliter"
                    elif lowest_level_pack_value <= 100.0:
                        train_df.at[i, 'pack_size'] = "LoMilliliter"
                    elif lowest_level_pack_value <= 750.0:
                        train_df.at[i, 'pack_size'] = "MedMilliliter"
                    elif lowest_level_pack_value > 750.0:
                        train_df.at[i, 'pack_size'] = "HiMilliliter"
                elif lowest_level_pack_uom == "ft" or lowest_level_pack_uom == "ftq" or \
                     lowest_level_pack_uom == "fot" or lowest_level_pack_uom == "foot" or \
                     lowest_level_pack_uom == "feet":
                    if lowest_level_pack_value <= 5.0:
                        train_df.at[i, 'pack_size'] = "LoFeet"
                    elif lowest_level_pack_value <= 100.0:
                        train_df.at[i, 'pack_size'] = "MedFeet"
                    elif lowest_level_pack_value > 100.0:
                        train_df.at[i, 'pack_size'] = "HiFeet"
                elif lowest_level_pack_uom == "gr" or lowest_level_pack_uom == "grm" or \
                        lowest_level_pack_uom == "gram":
                    if lowest_level_pack_value <= 5.0:
                        train_df.at[i, 'pack_size'] = "TinyGram"
                    elif lowest_level_pack_value <= 20.0:
                        train_df.at[i, 'pack_size'] = "LoGram"
                    elif lowest_level_pack_value <= 100.0:
                        train_df.at[i, 'pack_size'] = "MedGram"
                    elif lowest_level_pack_value > 100.0:
                        train_df.at[i, 'pack_size'] = "HiGram"
                elif lowest_level_pack_uom == "st" or lowest_level_pack_uom == "set":
                    train_df.at[i, 'pack_size'] = "Set"
                elif lowest_level_pack_uom == "bu" or lowest_level_pack_uom == "bushel":
                    train_df.at[i, 'pack_size'] = "Bushel"
                elif lowest_level_pack_uom == "pk" or lowest_level_pack_uom == "pack":
                    train_df.at[i, 'pack_size'] = "Pack"
                elif lowest_level_pack_uom == "pr" or lowest_level_pack_uom == "pair":
                    train_df.at[i, 'pack_size'] = "Pair"

                # Flag for multiple levels in the pack size
                if has_two_levels:
                    train_df.at[i, 'pack_size'] = train_df.at[i, 'pack_size'] + " withtwolevels"
                if has_three_levels:
                    train_df.at[i, 'pack_size'] = train_df.at[i, 'pack_size'] + " withthreelevels"

                has_two_levels = False
                has_three_levels = False

            except:
                train_df.at[i, 'pack_size'] = "UnrecognizedPacksize"
        except:
            train_df.at[i, 'pack_size'] = "UnrecognizedPacksize"

    # Converting pack sizes to generalized terms for validation dataset
    for i in tqdm(range(validation_df.shape[0])):
        orig_pack = str(validation_df.loc[i, 'pack_size'])
        try:

            has_two_levels = False
            has_three_levels = False

            # Parse the pack size
            slash_split_pack = orig_pack.split('/')  # parses by '/' and returns a list of the substrings

            if len(slash_split_pack) >= 3:
                has_three_levels = True
            elif len(slash_split_pack) == 2:
                has_two_levels = True

            lowest_level_pack = slash_split_pack[len(slash_split_pack) - 1]  # capture the last substring in the list
            whitespace_split_pack = lowest_level_pack.split()  # parses by whitespace and returns a list
            lowest_level_pack_value = whitespace_split_pack[0]  # index 0 s/b the value
            lowest_level_pack_uom = whitespace_split_pack[1]  # index 1 s/b the uom

            # Check for a dash (often used for ranges)
            dash_split = lowest_level_pack_value.split('-')
            if len(dash_split) > 1:
                lowest_level_pack_value = dash_split[len(dash_split) - 1]  # If there is more than one value, use the last one

            # Convert the lowest level pack size value to a float for comparisons below
            lowest_level_pack_value = float(lowest_level_pack_value)

            # Run logic below to convert pack size to general terms
            try:
                if lowest_level_pack_uom == "lb" or lowest_level_pack_uom == "lbr" or \
                        lowest_level_pack_uom == "lba" or lowest_level_pack_uom == "pound":
                    if lowest_level_pack_value <= 1.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto1lb"
                    elif lowest_level_pack_value <= 5.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto5lb"
                    elif lowest_level_pack_value <= 10.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto10lb"
                    elif lowest_level_pack_value <= 20.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto20lb"
                    elif lowest_level_pack_value <= 100.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto100lb"
                    elif lowest_level_pack_value > 100.0:
                        validation_df.at[i, 'pack_size'] = "greaterthan100lb"
                elif lowest_level_pack_uom == "oz" or lowest_level_pack_uom == "oza" or \
                     lowest_level_pack_uom == "onz" or lowest_level_pack_uom == "ounce" or \
                     lowest_level_pack_uom == "z" or lowest_level_pack_uom == "za":
                    if lowest_level_pack_value <= 1.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto1oz"
                    elif lowest_level_pack_value <= 2.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto2oz"
                    elif lowest_level_pack_value <= 3.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto3oz"
                    elif lowest_level_pack_value <= 4.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto4oz"
                    elif lowest_level_pack_value <= 5.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto5oz"
                    elif lowest_level_pack_value <= 6.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto6oz"
                    elif lowest_level_pack_value <= 8.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto8oz"
                    elif lowest_level_pack_value <= 12.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto12oz"
                    elif lowest_level_pack_value <= 16.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto16oz"
                    elif lowest_level_pack_value <= 20.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto20oz"
                    elif lowest_level_pack_value <= 24.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto24oz"
                    elif lowest_level_pack_value <= 32.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto32oz"
                    elif lowest_level_pack_value > 32.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto100oz"
                    elif lowest_level_pack_value <= 64.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto64oz"
                    elif lowest_level_pack_value <= 100.0:
                        validation_df.at[i, 'pack_size'] = "lessequalto100oz"
                    elif lowest_level_pack_value > 100.0:
                        validation_df.at[i, 'pack_size'] = "greaterthan100oz"
                elif lowest_level_pack_uom == "ea" or lowest_level_pack_uom == "ct" or \
                        lowest_level_pack_uom == "count" or lowest_level_pack_uom == "each":
                    if lowest_level_pack_value <= 1.0:
                        validation_df.at[i, 'pack_size'] = "oneeach"
                    elif lowest_level_pack_value <= 5.0:
                        validation_df.at[i, 'pack_size'] = "lessthan5ea"
                    elif lowest_level_pack_value <= 10.0:
                        validation_df.at[i, 'pack_size'] = "lessthan10ea"
                    elif lowest_level_pack_value <= 15.0:
                        validation_df.at[i, 'pack_size'] = "lessthan15ea"
                    elif lowest_level_pack_value <= 20.0:
                        validation_df.at[i, 'pack_size'] = "lessthan20ea"
                    elif lowest_level_pack_value <= 25.0:
                        validation_df.at[i, 'pack_size'] = "lessthan25ea"
                    elif lowest_level_pack_value <= 30.0:
                        validation_df.at[i, 'pack_size'] = "lessthan30ea"
                    elif lowest_level_pack_value <= 40.0:
                        validation_df.at[i, 'pack_size'] = "lessthan40ea"
                    elif lowest_level_pack_value <= 50.0:
                        validation_df.at[i, 'pack_size'] = "lessthan50ea"
                    elif lowest_level_pack_value <= 75.0:
                        validation_df.at[i, 'pack_size'] = "lessthan75ea"
                    elif lowest_level_pack_value <= 100.0:
                        validation_df.at[i, 'pack_size'] = "lessthan100ea"
                    elif lowest_level_pack_value <= 150.0:
                        validation_df.at[i, 'pack_size'] = "lessthan150ea"
                    elif lowest_level_pack_value <= 200.0:
                        validation_df.at[i, 'pack_size'] = "lessthan200ea"
                    elif lowest_level_pack_value <= 250.0:
                        validation_df.at[i, 'pack_size'] = "lessthan250ea"
                    elif lowest_level_pack_value <= 500.0:
                        validation_df.at[i, 'pack_size'] = "lessthan500ea"
                    elif lowest_level_pack_value <= 1000.0:
                        validation_df.at[i, 'pack_size'] = "lessthan1000ea"
                    elif lowest_level_pack_value > 1000.0:
                        validation_df.at[i, 'pack_size'] = "greaterthan1000ea"
                elif lowest_level_pack_uom == "dz" or lowest_level_pack_uom == "doz" or \
                        lowest_level_pack_uom == "dzn" or lowest_level_pack_uom == "dozen":
                    if lowest_level_pack_value <= 1.0:
                        validation_df.at[i, 'pack_size'] = "TinyDozen"
                    elif lowest_level_pack_value <= 3.0:
                        validation_df.at[i, 'pack_size'] = "LoDozen"
                    elif lowest_level_pack_value <= 6.0:
                        validation_df.at[i, 'pack_size'] = "LoMedDozen"
                    elif lowest_level_pack_value <= 20.0:
                        validation_df.at[i, 'pack_size'] = "MedDozen"
                    elif lowest_level_pack_value > 20.0:
                        validation_df.at[i, 'pack_size'] = "HiDozen"
                elif lowest_level_pack_uom == "ga" or lowest_level_pack_uom == "gll" or \
                        lowest_level_pack_uom == "gal" or lowest_level_pack_uom == "gallon":
                    if lowest_level_pack_value <= 1.0:
                        validation_df.at[i, 'pack_size'] = "TinyGallon"
                    elif lowest_level_pack_value <= 3.0:
                        validation_df.at[i, 'pack_size'] = "LoGallon"
                    elif lowest_level_pack_value <= 5.0:
                        validation_df.at[i, 'pack_size'] = "MedGallon"
                    elif lowest_level_pack_value > 5.0:
                        validation_df.at[i, 'pack_size'] = "HiGallon"
                elif lowest_level_pack_uom == "rl" or lowest_level_pack_uom == "roll":
                    if lowest_level_pack_value <= 10.0:
                        validation_df.at[i, 'pack_size'] = "LoRoll"
                    elif lowest_level_pack_value <= 50.0:
                        validation_df.at[i, 'pack_size'] = "MedRoll"
                    elif lowest_level_pack_value > 50.0:
                        validation_df.at[i, 'pack_size'] = "HiRoll"
                elif lowest_level_pack_uom == "cn" or lowest_level_pack_uom == "can":
                    validation_df.at[i, 'pack_size'] = "Can"
                elif lowest_level_pack_uom == "lt" or lowest_level_pack_uom == "ltr" or \
                        lowest_level_pack_uom == "liter" or lowest_level_pack_uom == "litre":
                    if lowest_level_pack_value <= 1.0:
                        validation_df.at[i, 'pack_size'] = "TinyLiter"
                    elif lowest_level_pack_value <= 4.0:
                        validation_df.at[i, 'pack_size'] = "LoLiter"
                    elif lowest_level_pack_value <= 12.0:
                        validation_df.at[i, 'pack_size'] = "MedLiter"
                    elif lowest_level_pack_value > 12.0:
                        validation_df.at[i, 'pack_size'] = "HiLiter"
                elif lowest_level_pack_uom == "kg" or lowest_level_pack_uom == "kgr" or \
                     lowest_level_pack_uom == "kgm" or lowest_level_pack_uom == "kilo" or \
                     lowest_level_pack_uom == "kilogram":
                    if lowest_level_pack_value <= 3.0:
                        validation_df.at[i, 'pack_size'] = "LoKilogram"
                    elif lowest_level_pack_value <= 12.0:
                        validation_df.at[i, 'pack_size'] = "MedKilogram"
                    elif lowest_level_pack_value > 12.0:
                        validation_df.at[i, 'pack_size'] = "HiKilogram"
                elif lowest_level_pack_uom == "qt" or lowest_level_pack_uom == "qtl" or \
                        lowest_level_pack_uom == "qtd" or lowest_level_pack_uom == "quart":
                    if lowest_level_pack_value <= 4.0:
                        validation_df.at[i, 'pack_size'] = "LoQuart"
                    elif lowest_level_pack_value <= 8.0:
                        validation_df.at[i, 'pack_size'] = "MedQuart"
                    elif lowest_level_pack_value > 8.0:
                        validation_df.at[i, 'pack_size'] = "HiQuart"
                elif lowest_level_pack_uom == "pt" or lowest_level_pack_uom == "ptl" or \
                        lowest_level_pack_uom == "pti" or lowest_level_pack_uom == "pint":
                    if lowest_level_pack_value <= 1.0:
                        validation_df.at[i, 'pack_size'] = "LoPint"
                    elif lowest_level_pack_value <= 4.0:
                        validation_df.at[i, 'pack_size'] = "MedPint"
                    elif lowest_level_pack_value > 4.0:
                        validation_df.at[i, 'pack_size'] = "HiPint"
                elif lowest_level_pack_uom == "kt" or lowest_level_pack_uom == "kit":
                    validation_df.at[i, 'pack_size'] = "Kit"
                elif lowest_level_pack_uom == "ml" or lowest_level_pack_uom == "mlt" or \
                        lowest_level_pack_uom == "milliliter" or lowest_level_pack_uom == "millilitre":
                    if lowest_level_pack_value <= 25.0:
                        validation_df.at[i, 'pack_size'] = "TinyMilliliter"
                    elif lowest_level_pack_value <= 100.0:
                        validation_df.at[i, 'pack_size'] = "LoMilliliter"
                    elif lowest_level_pack_value <= 750.0:
                        validation_df.at[i, 'pack_size'] = "MedMilliliter"
                    elif lowest_level_pack_value > 750.0:
                        validation_df.at[i, 'pack_size'] = "HiMilliliter"
                elif lowest_level_pack_uom == "ft" or lowest_level_pack_uom == "ftq" or \
                     lowest_level_pack_uom == "fot" or lowest_level_pack_uom == "foot" or \
                     lowest_level_pack_uom == "feet":
                    if lowest_level_pack_value <= 5.0:
                        validation_df.at[i, 'pack_size'] = "LoFeet"
                    elif lowest_level_pack_value <= 100.0:
                        validation_df.at[i, 'pack_size'] = "MedFeet"
                    elif lowest_level_pack_value > 100.0:
                        validation_df.at[i, 'pack_size'] = "HiFeet"
                elif lowest_level_pack_uom == "gr" or lowest_level_pack_uom == "grm" or \
                        lowest_level_pack_uom == "gram":
                    if lowest_level_pack_value <= 5.0:
                        validation_df.at[i, 'pack_size'] = "TinyGram"
                    elif lowest_level_pack_value <= 20.0:
                        validation_df.at[i, 'pack_size'] = "LoGram"
                    elif lowest_level_pack_value <= 100.0:
                        validation_df.at[i, 'pack_size'] = "MedGram"
                    elif lowest_level_pack_value > 100.0:
                        validation_df.at[i, 'pack_size'] = "HiGram"
                elif lowest_level_pack_uom == "st" or lowest_level_pack_uom == "set":
                    validation_df.at[i, 'pack_size'] = "Set"
                elif lowest_level_pack_uom == "bu" or lowest_level_pack_uom == "bushel":
                    validation_df.at[i, 'pack_size'] = "Bushel"
                elif lowest_level_pack_uom == "pk" or lowest_level_pack_uom == "pack":
                    validation_df.at[i, 'pack_size'] = "Pack"
                elif lowest_level_pack_uom == "pr" or lowest_level_pack_uom == "pair":
                    validation_df.at[i, 'pack_size'] = "Pair"

                # Flag for multiple levels in the pack size
                if has_two_levels:
                    validation_df.at[i, 'pack_size'] = validation_df.at[i, 'pack_size'] + " withtwolevels"
                if has_three_levels:
                    validation_df.at[i, 'pack_size'] = validation_df.at[i, 'pack_size'] + " withthreelevels"

                has_two_levels = False
                has_three_levels = False

            except:
                validation_df.at[i, 'pack_size'] = "UnrecognizedPacksize"
        except:
            validation_df.at[i, 'pack_size'] = "UnrecognizedPacksize"

    # Converting pack sizes to generalized terms for test dataset
    for i in tqdm(range(test_df.shape[0])):
        orig_pack = str(test_df.loc[i, 'pack_size'])
        try:

            has_two_levels = False
            has_three_levels = False

            # Parse the pack size
            slash_split_pack = orig_pack.split('/')  # parses by '/' and returns a list of the substrings

            if len(slash_split_pack) >= 3:
                has_three_levels = True
            elif len(slash_split_pack) == 2:
                has_two_levels = True

            lowest_level_pack = slash_split_pack[len(slash_split_pack) - 1]  # capture the last substring in the list
            whitespace_split_pack = lowest_level_pack.split()  # parses by whitespace and returns a list
            lowest_level_pack_value = whitespace_split_pack[0]  # index 0 s/b the value
            lowest_level_pack_uom = whitespace_split_pack[1]  # index 1 s/b the uom

            # Check for a dash (often used for ranges)
            dash_split = lowest_level_pack_value.split('-')
            if len(dash_split) > 1:
                lowest_level_pack_value = dash_split[len(dash_split) - 1]  # If there is more than one value, use the last one

            # Convert the lowest level pack size value to a float for comparisons below
            lowest_level_pack_value = float(lowest_level_pack_value)

            # Run logic below to convert pack size to general terms
            try:
                if lowest_level_pack_uom == "lb" or lowest_level_pack_uom == "lbr" or \
                        lowest_level_pack_uom == "lba" or lowest_level_pack_uom == "pound":
                    if lowest_level_pack_value <= 1.0:
                        test_df.at[i, 'pack_size'] = "lessequalto1lb"
                    elif lowest_level_pack_value <= 5.0:
                        test_df.at[i, 'pack_size'] = "lessequalto5lb"
                    elif lowest_level_pack_value <= 10.0:
                        test_df.at[i, 'pack_size'] = "lessequalto10lb"
                    elif lowest_level_pack_value <= 20.0:
                        test_df.at[i, 'pack_size'] = "lessequalto20lb"
                    elif lowest_level_pack_value <= 100.0:
                        test_df.at[i, 'pack_size'] = "lessequalto100lb"
                    elif lowest_level_pack_value > 100.0:
                        test_df.at[i, 'pack_size'] = "greaterthan100lb"
                elif lowest_level_pack_uom == "oz" or lowest_level_pack_uom == "oza" or \
                     lowest_level_pack_uom == "onz" or lowest_level_pack_uom == "ounce" or \
                     lowest_level_pack_uom == "z" or lowest_level_pack_uom == "za":
                    if lowest_level_pack_value <= 1.0:
                        test_df.at[i, 'pack_size'] = "lessequalto1oz"
                    elif lowest_level_pack_value <= 2.0:
                        test_df.at[i, 'pack_size'] = "lessequalto2oz"
                    elif lowest_level_pack_value <= 3.0:
                        test_df.at[i, 'pack_size'] = "lessequalto3oz"
                    elif lowest_level_pack_value <= 4.0:
                        test_df.at[i, 'pack_size'] = "lessequalto4oz"
                    elif lowest_level_pack_value <= 5.0:
                        test_df.at[i, 'pack_size'] = "lessequalto5oz"
                    elif lowest_level_pack_value <= 6.0:
                        test_df.at[i, 'pack_size'] = "lessequalto6oz"
                    elif lowest_level_pack_value <= 8.0:
                        test_df.at[i, 'pack_size'] = "lessequalto8oz"
                    elif lowest_level_pack_value <= 12.0:
                        test_df.at[i, 'pack_size'] = "lessequalto12oz"
                    elif lowest_level_pack_value <= 16.0:
                        test_df.at[i, 'pack_size'] = "lessequalto16oz"
                    elif lowest_level_pack_value <= 20.0:
                        test_df.at[i, 'pack_size'] = "lessequalto20oz"
                    elif lowest_level_pack_value <= 24.0:
                        test_df.at[i, 'pack_size'] = "lessequalto24oz"
                    elif lowest_level_pack_value <= 32.0:
                        test_df.at[i, 'pack_size'] = "lessequalto32oz"
                    elif lowest_level_pack_value > 32.0:
                        test_df.at[i, 'pack_size'] = "lessequalto100oz"
                    elif lowest_level_pack_value <= 64.0:
                        test_df.at[i, 'pack_size'] = "lessequalto64oz"
                    elif lowest_level_pack_value <= 100.0:
                        test_df.at[i, 'pack_size'] = "lessequalto100oz"
                    elif lowest_level_pack_value > 100.0:
                        test_df.at[i, 'pack_size'] = "greaterthan100oz"
                elif lowest_level_pack_uom == "ea" or lowest_level_pack_uom == "ct" or \
                        lowest_level_pack_uom == "count" or lowest_level_pack_uom == "each":
                    if lowest_level_pack_value <= 1.0:
                        test_df.at[i, 'pack_size'] = "oneeach"
                    elif lowest_level_pack_value <= 5.0:
                        test_df.at[i, 'pack_size'] = "lessthan5ea"
                    elif lowest_level_pack_value <= 10.0:
                        test_df.at[i, 'pack_size'] = "lessthan10ea"
                    elif lowest_level_pack_value <= 15.0:
                        test_df.at[i, 'pack_size'] = "lessthan15ea"
                    elif lowest_level_pack_value <= 20.0:
                        test_df.at[i, 'pack_size'] = "lessthan20ea"
                    elif lowest_level_pack_value <= 25.0:
                        test_df.at[i, 'pack_size'] = "lessthan25ea"
                    elif lowest_level_pack_value <= 30.0:
                        test_df.at[i, 'pack_size'] = "lessthan30ea"
                    elif lowest_level_pack_value <= 40.0:
                        test_df.at[i, 'pack_size'] = "lessthan40ea"
                    elif lowest_level_pack_value <= 50.0:
                        test_df.at[i, 'pack_size'] = "lessthan50ea"
                    elif lowest_level_pack_value <= 75.0:
                        test_df.at[i, 'pack_size'] = "lessthan75ea"
                    elif lowest_level_pack_value <= 100.0:
                        test_df.at[i, 'pack_size'] = "lessthan100ea"
                    elif lowest_level_pack_value <= 150.0:
                        test_df.at[i, 'pack_size'] = "lessthan150ea"
                    elif lowest_level_pack_value <= 200.0:
                        test_df.at[i, 'pack_size'] = "lessthan200ea"
                    elif lowest_level_pack_value <= 250.0:
                        test_df.at[i, 'pack_size'] = "lessthan250ea"
                    elif lowest_level_pack_value <= 500.0:
                        test_df.at[i, 'pack_size'] = "lessthan500ea"
                    elif lowest_level_pack_value <= 1000.0:
                        test_df.at[i, 'pack_size'] = "lessthan1000ea"
                    elif lowest_level_pack_value > 1000.0:
                        test_df.at[i, 'pack_size'] = "greaterthan1000ea"
                elif lowest_level_pack_uom == "dz" or lowest_level_pack_uom == "doz" or \
                        lowest_level_pack_uom == "dzn" or lowest_level_pack_uom == "dozen":
                    if lowest_level_pack_value <= 1.0:
                        test_df.at[i, 'pack_size'] = "TinyDozen"
                    elif lowest_level_pack_value <= 3.0:
                        test_df.at[i, 'pack_size'] = "LoDozen"
                    elif lowest_level_pack_value <= 6.0:
                        test_df.at[i, 'pack_size'] = "LoMedDozen"
                    elif lowest_level_pack_value <= 20.0:
                        test_df.at[i, 'pack_size'] = "MedDozen"
                    elif lowest_level_pack_value > 20.0:
                        test_df.at[i, 'pack_size'] = "HiDozen"
                elif lowest_level_pack_uom == "ga" or lowest_level_pack_uom == "gll" or \
                        lowest_level_pack_uom == "gal" or lowest_level_pack_uom == "gallon":
                    if lowest_level_pack_value <= 1.0:
                        test_df.at[i, 'pack_size'] = "TinyGallon"
                    elif lowest_level_pack_value <= 3.0:
                        test_df.at[i, 'pack_size'] = "LoGallon"
                    elif lowest_level_pack_value <= 5.0:
                        test_df.at[i, 'pack_size'] = "MedGallon"
                    elif lowest_level_pack_value > 5.0:
                        test_df.at[i, 'pack_size'] = "HiGallon"
                elif lowest_level_pack_uom == "rl" or lowest_level_pack_uom == "roll":
                    if lowest_level_pack_value <= 10.0:
                        test_df.at[i, 'pack_size'] = "LoRoll"
                    elif lowest_level_pack_value <= 50.0:
                        test_df.at[i, 'pack_size'] = "MedRoll"
                    elif lowest_level_pack_value > 50.0:
                        test_df.at[i, 'pack_size'] = "HiRoll"
                elif lowest_level_pack_uom == "cn" or lowest_level_pack_uom == "can":
                    test_df.at[i, 'pack_size'] = "Can"
                elif lowest_level_pack_uom == "lt" or lowest_level_pack_uom == "ltr" or \
                        lowest_level_pack_uom == "liter" or lowest_level_pack_uom == "litre":
                    if lowest_level_pack_value <= 1.0:
                        test_df.at[i, 'pack_size'] = "TinyLiter"
                    elif lowest_level_pack_value <= 4.0:
                        test_df.at[i, 'pack_size'] = "LoLiter"
                    elif lowest_level_pack_value <= 12.0:
                        test_df.at[i, 'pack_size'] = "MedLiter"
                    elif lowest_level_pack_value > 12.0:
                        test_df.at[i, 'pack_size'] = "HiLiter"
                elif lowest_level_pack_uom == "kg" or lowest_level_pack_uom == "kgr" or \
                     lowest_level_pack_uom == "kgm" or lowest_level_pack_uom == "kilo" or \
                     lowest_level_pack_uom == "kilogram":
                    if lowest_level_pack_value <= 3.0:
                        test_df.at[i, 'pack_size'] = "LoKilogram"
                    elif lowest_level_pack_value <= 12.0:
                        test_df.at[i, 'pack_size'] = "MedKilogram"
                    elif lowest_level_pack_value > 12.0:
                        test_df.at[i, 'pack_size'] = "HiKilogram"
                elif lowest_level_pack_uom == "qt" or lowest_level_pack_uom == "qtl" or \
                        lowest_level_pack_uom == "qtd" or lowest_level_pack_uom == "quart":
                    if lowest_level_pack_value <= 4.0:
                        test_df.at[i, 'pack_size'] = "LoQuart"
                    elif lowest_level_pack_value <= 8.0:
                        test_df.at[i, 'pack_size'] = "MedQuart"
                    elif lowest_level_pack_value > 8.0:
                        test_df.at[i, 'pack_size'] = "HiQuart"
                elif lowest_level_pack_uom == "pt" or lowest_level_pack_uom == "ptl" or \
                        lowest_level_pack_uom == "pti" or lowest_level_pack_uom == "pint":
                    if lowest_level_pack_value <= 1.0:
                        test_df.at[i, 'pack_size'] = "LoPint"
                    elif lowest_level_pack_value <= 4.0:
                        test_df.at[i, 'pack_size'] = "MedPint"
                    elif lowest_level_pack_value > 4.0:
                        test_df.at[i, 'pack_size'] = "HiPint"
                elif lowest_level_pack_uom == "kt" or lowest_level_pack_uom == "kit":
                    test_df.at[i, 'pack_size'] = "Kit"
                elif lowest_level_pack_uom == "ml" or lowest_level_pack_uom == "mlt" or \
                        lowest_level_pack_uom == "milliliter" or lowest_level_pack_uom == "millilitre":
                    if lowest_level_pack_value <= 25.0:
                        test_df.at[i, 'pack_size'] = "TinyMilliliter"
                    elif lowest_level_pack_value <= 100.0:
                        test_df.at[i, 'pack_size'] = "LoMilliliter"
                    elif lowest_level_pack_value <= 750.0:
                        test_df.at[i, 'pack_size'] = "MedMilliliter"
                    elif lowest_level_pack_value > 750.0:
                        test_df.at[i, 'pack_size'] = "HiMilliliter"
                elif lowest_level_pack_uom == "ft" or lowest_level_pack_uom == "ftq" or \
                     lowest_level_pack_uom == "fot" or lowest_level_pack_uom == "foot" or \
                     lowest_level_pack_uom == "feet":
                    if lowest_level_pack_value <= 5.0:
                        test_df.at[i, 'pack_size'] = "LoFeet"
                    elif lowest_level_pack_value <= 100.0:
                        test_df.at[i, 'pack_size'] = "MedFeet"
                    elif lowest_level_pack_value > 100.0:
                        test_df.at[i, 'pack_size'] = "HiFeet"
                elif lowest_level_pack_uom == "gr" or lowest_level_pack_uom == "grm" or \
                        lowest_level_pack_uom == "gram":
                    if lowest_level_pack_value <= 5.0:
                        test_df.at[i, 'pack_size'] = "TinyGram"
                    elif lowest_level_pack_value <= 20.0:
                        test_df.at[i, 'pack_size'] = "LoGram"
                    elif lowest_level_pack_value <= 100.0:
                        test_df.at[i, 'pack_size'] = "MedGram"
                    elif lowest_level_pack_value > 100.0:
                        test_df.at[i, 'pack_size'] = "HiGram"
                elif lowest_level_pack_uom == "st" or lowest_level_pack_uom == "set":
                    test_df.at[i, 'pack_size'] = "Set"
                elif lowest_level_pack_uom == "bu" or lowest_level_pack_uom == "bushel":
                    test_df.at[i, 'pack_size'] = "Bushel"
                elif lowest_level_pack_uom == "pk" or lowest_level_pack_uom == "pack":
                    test_df.at[i, 'pack_size'] = "Pack"
                elif lowest_level_pack_uom == "pr" or lowest_level_pack_uom == "pair":
                    test_df.at[i, 'pack_size'] = "Pair"

                # Flag for multiple levels in the pack size
                if has_two_levels:
                    test_df.at[i, 'pack_size'] = test_df.at[i, 'pack_size'] + " withtwolevels"
                if has_three_levels:
                    test_df.at[i, 'pack_size'] = test_df.at[i, 'pack_size'] + " withthreelevels"

            except:
                test_df.at[i, 'pack_size'] = "UnrecognizedPacksize"
        except:
            test_df.at[i, 'pack_size'] = "UnrecognizedPacksize"

    #####
    # Converting storage temperatures to storage descriptors
    #

    print("\n Converting storage temps to storage terms...\n")

    time.sleep(.3)  # brief delay to manage tqdm

    # Converting the Storage Temps for the Training Dataset
    for i in tqdm(range(train_df.shape[0])):

        min_temp = train_df.loc[i, 'temp_min']
        max_temp = train_df.loc[i, 'temp_max']
        try:
            min_temp = int(min_temp)
        except:
            min_temp = 10000
        try:
            max_temp = int(max_temp)
        except:
            max_temp = -10000
        if int(min_temp) <= 15 and int(max_temp) <= 32:
            desc = train_df.loc[i, 'description']
            if "frozen" not in desc:
                train_df.at[i, 'description'] = desc + " frozen"
        elif int(max_temp) < 28:
            desc = train_df.loc[i, 'description']
            if "frozen" not in desc:
                train_df.at[i, 'description'] = desc + " frozen"
        # Logic for ref storage
        elif int(max_temp) <= 45:
            desc = train_df.loc[i, 'description']
            if "ref" not in desc:
                train_df.at[i, 'description'] = desc + " ref"
        # Logic for upper ref storage
        elif int(max_temp) <= 55:
            desc = train_df.loc[i, 'description']
            if "ref" not in desc:
                train_df.at[i, 'description'] = desc + " upper ref"
            elif "upper" not in desc:
                train_df.at[i, 'description'] = desc + " upper"
        # Logic for upperupper ref storage
        elif int(max_temp) <= 70:
            desc = train_df.loc[i, 'description']
            if "ref" not in desc:
                train_df.at[i, 'description'] = desc + " upperupper ref"
            elif "upperupper" not in desc:
                train_df.at[i, 'description'] = desc + " upperupper"
        # Logic for shelfstable storage
        elif int(max_temp) >= 71:
            desc = train_df.loc[i, 'description']
            if "shelfstable" not in desc:
                train_df.at[i, 'description'] = desc + " shelfstable"

    # Converting the Storage Temps for the Validation Dataset
    for i in tqdm(range(validation_df.shape[0])):

        min_temp = validation_df.loc[i, 'temp_min']
        max_temp = validation_df.loc[i, 'temp_max']
        try:
            min_temp = int(min_temp)
        except:
            min_temp = 10000
        try:
            max_temp = int(max_temp)
        except:
            max_temp = -10000
        # Logic for frozen storage
        if int(min_temp) <= 15 and int(max_temp) <= 32:
            desc = validation_df.loc[i, 'description']
            if "frozen" not in desc:
                validation_df.at[i, 'description'] = desc + " frozen"
        elif int(max_temp) < 28:
            desc = validation_df.loc[i, 'description']
            if "frozen" not in desc:
                validation_df.at[i, 'description'] = desc + " frozen"
        # Logic for ref storage
        # Logic for ref storage
        elif int(max_temp) <= 45:
            desc = validation_df.loc[i, 'description']
            if "ref" not in desc:
                validation_df.at[i, 'description'] = desc + " ref"
        # Logic for upper ref storage
        elif int(max_temp) <= 55:
            desc = validation_df.loc[i, 'description']
            if "ref" not in desc:
                validation_df.at[i, 'description'] = desc + " upper ref"
            elif "upper" not in desc:
                validation_df.at[i, 'description'] = desc + " upper"
        # Logic for upperupper ref storage
        elif int(max_temp) <= 70:
            desc = validation_df.loc[i, 'description']
            if "ref" not in desc:
                validation_df.at[i, 'description'] = desc + " upperupper ref"
            elif "upperupper" not in desc:
                validation_df.at[i, 'description'] = desc + " upperupper"
        # Logic for shelfstable storage
        elif int(max_temp) >= 71:
            desc = validation_df.loc[i, 'description']
            if "shelfstable" not in desc:
                validation_df.at[i, 'description'] = desc + " shelfstable"
    # Converting the Storage Temps for the Test Dataset
    for i in tqdm(range(test_df.shape[0])):

        min_temp = test_df.loc[i, 'temp_min']
        max_temp = test_df.loc[i, 'temp_max']
        try:
            min_temp = int(min_temp)
        except:
            min_temp = 10000
        try:
            max_temp = int(max_temp)
        except:
            max_temp = -10000
        # Logic for frozen storage
        if int(min_temp) <= 15 and int(max_temp) <= 32:
            desc = test_df.loc[i, 'description']
            if "frozen" not in desc:
                test_df.at[i, 'description'] = desc + " frozen"
        elif int(max_temp) < 28:
            desc = test_df.loc[i, 'description']
            if "frozen" not in desc:
                test_df.at[i, 'description'] = desc + " frozen"
        # Logic for ref storage
        elif int(max_temp) <= 45:
            desc = test_df.loc[i, 'description']
            if "ref" not in desc:
                test_df.at[i, 'description'] = desc + " ref"
        # Logic for upper ref storage
        elif int(max_temp) <= 55:
            desc = test_df.loc[i, 'description']
            if "ref" not in desc:
                test_df.at[i, 'description'] = desc + " upper ref"
            elif "upper" not in desc:
                test_df.at[i, 'description'] = desc + " upper"
        # Logic for upperupper ref storage
        elif int(max_temp) <= 70:
            desc = test_df.loc[i, 'description']
            if "ref" not in desc:
                test_df.at[i, 'description'] = desc + " upperupper ref"
            elif "upperupper" not in desc:
                test_df.at[i, 'description'] = desc + " upperupper"
        # Logic for shelfstable storage
        elif int(max_temp) >= 71:
            desc = test_df.loc[i, 'description']
            if "shelfstable" not in desc:
                test_df.at[i, 'description'] = desc + " shelfstable"

    #####
    # Concatenating the description, coo, and pack_size into a single text field
    #

    # Concatenating description-related fields for the Training Dataset
    train_df['description'] = train_df.description.astype(str).str.cat(train_df.coo.astype(str), sep=' ')
    train_df['description'] = train_df.description.astype(str).str.cat(train_df.pack_size.astype(str), sep=' ')
    train_df['description'] = train_df.description.astype(str).str.cat(train_df.brand.astype(str), sep=' ')
    train_df['description'] = train_df.description.astype(str).str.cat(train_df.manufacturer.astype(str), sep=' ')
    train_df = prune_data_final(train_df)

    # Concatenating description-related fields for the Validation Dataset
    validation_df['description'] = validation_df.description.astype(str).str.cat(validation_df.coo.astype(str), sep=' ')
    validation_df['description'] = validation_df.description.astype(str).str.cat(validation_df.pack_size.astype(str), sep=' ')
    validation_df['description'] = validation_df.description.astype(str).str.cat(validation_df.brand.astype(str), sep=' ')
    validation_df['description'] = validation_df.description.astype(str).str.cat(validation_df.manufacturer.astype(str), sep=' ')
    validation_df = prune_data_final(validation_df)

    # Concatenating description-related fields for the Test Dataset
    test_df['description'] = test_df.description.astype(str).str.cat(test_df.coo.astype(str), sep=' ')
    test_df['description'] = test_df.description.astype(str).str.cat(test_df.pack_size.astype(str), sep=' ')
    test_df['description'] = test_df.description.astype(str).str.cat(test_df.brand.astype(str), sep=' ')
    test_df['description'] = test_df.description.astype(str).str.cat(test_df.manufacturer.astype(str), sep=' ')
    test_df = prune_data_final(test_df)

    train_df['description'] = train_df['description'].apply(clean_none)
    validation_df['description'] = validation_df['description'].apply(clean_none)
    test_df['description'] = test_df['description'].apply(clean_none)

    return train_df, validation_df, test_df
