"""
DATA PREPROCESSOR

Developed By: Thomas Vaughn
Version: 1.1.4
Last Update Date: 7/10/2025

This script is designed to be imported as a module from the
Product Classifier script.

The prepare_dataset function prepares a dataset for training a
Product Classifier model.  It performs several specification
translation and generalization activities which support the
Product Classification task including:

    * Validating and Identifying input files for multiple use cases
    * Converting Category IDs to Classifier Labels
    * Converting pack sizes to generalized terms
    * Converting storage temperatures to generalized storage terms

This file contains the following functions:

    * prepare_dataset - Prepares data from an input file for the Product Classifier
    * convert_packsize - Converts packsize data from the Product Classifier's GUI

"""
import ctypes
from datasets import load_dataset, Dataset
import pandas as pd
import time
from data_cleaner import clean_data, clean_first_pass, clean_none, prune_data_final

pd.set_option('max_colwidth', 175)


def prepare_dataset(source_fpath):
    """

    :param source_fpath:
    :return:
    """

    # check to see if input file is .xlsx or .csv
    split_fname = source_fpath.split(".")
    file_extension = split_fname[len(split_fname) - 1]

    is_usecase_1 = False
    is_usecase_2 = False
    is_usecase_3 = False

    if file_extension == "csv":
        try:
            # Read the data into a pandas dataframe
            raw_data = pd.read_csv(source_fpath, encoding='utf-8', dtype=str)
        except:
            print("invalid input file")
            return "invalid_input_file", is_usecase_1, is_usecase_2, is_usecase_3
    elif file_extension == "xlsm":
        # Read the data into a pandas dataframe
        try:
            # Check for worksheet called "use_case_1"
            raw_data = pd.read_excel(source_fpath, sheet_name="use_case_1", skiprows=[0, 1, 2])
            is_usecase_1 = True
        except:
            try:
                # Check for worksheet called "use_case_2"
                raw_data = pd.read_excel(source_fpath, sheet_name="use_case_2", skiprows=[0])
                is_usecase_2 = True
            except:
                print("invalid_input_file")
                return "invalid_input_file", is_usecase_1, is_usecase_2, is_usecase_3
    elif file_extension == "xlsx":
        # Read the data into a pandas dataframe
        try:
            # Check for worksheet called "use_case_3"
            raw_data = pd.read_excel(source_fpath, sheet_name="use_case_3")
            is_usecase_3 = True
        except:
            print("invalid_input_file")
            return "invalid_input_file", is_usecase_1, is_usecase_2, is_usecase_3
    else:
        print("invalid input file")
        return "invalid_input_file", is_usecase_1, is_usecase_2, is_usecase_3

    # Strip the file extension from the filepath
    len_stripped_fpath = len(source_fpath) - 4
    stripped_fpath = source_fpath[:len_stripped_fpath]

    # Build the prepped data filepath
    prepped_fpath = "./Output_Files/_prepped.csv"

    # The logic for CSV files is in place for testing purposes
    if file_extension == "csv":

        # Run the data through first_pass cleaning run to prep the data
        prepped_data = clean_first_pass(raw_data)

        # Save the prepped data to a csv file
        prepped_data.to_csv(prepped_fpath, sep=',', encoding='utf-8', index=False)

        # Load the first_pass cleansed data into a huggingface dataset object
        dataset = load_dataset(
            'csv',
            data_files=prepped_fpath,
            delimiter=',',
            skiprows=1,
            column_names=['product_number', 'description', 'manufacturer', 'brand', 'coo',
                          'pack_size', 'temp_min', 'temp_min_uom', 'temp_max', 'temp_max_uom']
        )

        # Converting dataset to pandas format to prepare features for training
        dataset.set_format('pandas')
        # The dataset dict object returned from load_dataset places the primary dataset in ['train']
        raw_df = dataset['train'][:]
        # Run the data through the full data cleaning process
        df = clean_data(raw_df)

    elif file_extension == "xlsm" or file_extension == "xlsx":
        # Load the first_pass cleansed data into a huggingface dataset object
        file_is_valid = True
    else:
        print("invalid_input_file")
        return "invalid_input_file", is_usecase_1, is_usecase_2, is_usecase_3

    # clear the console
    # os.system('cls')
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 6)

    #####
    # Converting dataset to pandas format to prepare features for training
    #

    if file_extension == "xlsm" or file_extension == "xlsx":

        if is_usecase_1:

            # check to make sure the file contains the necessary headers
            try:
                raw_data['Manufacturer Name']
                raw_data['Brand Name']
                raw_data['Product Description']
                raw_data['Country Of Origin - Grown/Harvested']
                raw_data['Sales Pack Size Long']
                raw_data['Storage']
            except:
                print("missing_headers")
                return "missing_headers", is_usecase_1, is_usecase_2, is_usecase_3

            raw_data.rename(columns={'Manufacturer Name': 'manufacturer', 'Brand Name': 'brand',
                                     'Product Description': 'description', 'Country Of Origin - Grown/Harvested': 'coo',
                                     'Sales Pack Size Long': 'pack_size'}, inplace=True)

            # Run the data through first_pass cleaning run to prep the data
            prepped_data = clean_first_pass(raw_data)

            time.sleep(3)

            # Run the data through the full data cleaning process
            df = clean_data(prepped_data)

        elif is_usecase_2:

            # check to make sure the file contains the necessary headers
            try:
                raw_data['Manufacturer Name']
                raw_data['Brand Name']
                raw_data['Product Description']
                raw_data['Country Of Origin - Grown/Harvested']
                raw_data['Sales Pack Size Long']
                raw_data['Product Storage Code']
            except:
                print("missing_headers")
                return "missing_headers", is_usecase_1, is_usecase_2, is_usecase_3

            raw_data.rename(columns={'Manufacturer Name': 'manufacturer', 'Brand Name': 'brand',
                                     'Product Description': 'description',
                                     'Country Of Origin - Grown/Harvested': 'coo',
                                     'Sales Pack Size Long': 'pack_size'}, inplace=True)

            # Run the data through first_pass cleaning run to prep the data
            prepped_data = clean_first_pass(raw_data)

            # Run the data through the full data cleaning process
            df = clean_data(prepped_data)

        elif is_usecase_3:

            # check to make sure the file contains the necessary headers
            try:
                raw_data['Vendor Name']
                raw_data['Brand Name']
                raw_data['Product Description']
                raw_data['Country Of Origin - Grown/Harvested']
                raw_data['Sales Pack Size Long']
                raw_data['Storage Code']
                raw_data['GTIN']
                raw_data['Manufacturer Product Number']
            except:
                print("missing_headers")
                return "missing_headers", is_usecase_1, is_usecase_2, is_usecase_3

            raw_data.rename(columns={'Vendor Name': 'manufacturer', 'Brand Name': 'brand',
                                     'Country Of Origin - Grown/Harvested': 'coo',
                                     'Sales Pack Size Long': 'pack_size'}, inplace=True)

            # Run the data through first_pass cleaning run to prep the data
            prepped_data = clean_first_pass(raw_data)

            # Run the data through the full data cleaning process
            df = clean_data(prepped_data)

    elif file_extension == "csv":
        file_is_valid = True

    else:
        print("invalid_input_file")
        return "invalid_input_file", is_usecase_1, is_usecase_2, is_usecase_3

    #####
    # Converting pack sizes to general terms more easily managed by the model
    #

    time.sleep(.3)

    # Converting pack sizes to generalized terms
    for i in range(df.shape[0]):
        orig_pack = str(df.loc[i, 'pack_size'])

        # Check to see if uom is "can"
        if "can" in orig_pack:
            df.at[i, 'pack_size'] = "Can"
            continue

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
                        df.at[i, 'pack_size'] = "lessequalto1lb"
                    elif lowest_level_pack_value <= 5.0:
                        df.at[i, 'pack_size'] = "lessequalto5lb"
                    elif lowest_level_pack_value <= 10.0:
                        df.at[i, 'pack_size'] = "lessequalto10lb"
                    elif lowest_level_pack_value <= 20.0:
                        df.at[i, 'pack_size'] = "lessequalto20lb"
                    elif lowest_level_pack_value <= 100.0:
                        df.at[i, 'pack_size'] = "lessequalto100lb"
                    elif lowest_level_pack_value > 100.0:
                        df.at[i, 'pack_size'] = "greaterthan100lb"
                elif lowest_level_pack_uom == "oz" or lowest_level_pack_uom == "oza" or \
                        lowest_level_pack_uom == "onz" or lowest_level_pack_uom == "ounce":
                    if lowest_level_pack_value <= 1.0:
                        df.at[i, 'pack_size'] = "lessequalto1oz"
                    elif lowest_level_pack_value <= 2.0:
                        df.at[i, 'pack_size'] = "lessequalto2oz"
                    elif lowest_level_pack_value <= 3.0:
                        df.at[i, 'pack_size'] = "lessequalto3oz"
                    elif lowest_level_pack_value <= 4.0:
                        df.at[i, 'pack_size'] = "lessequalto4oz"
                    elif lowest_level_pack_value <= 5.0:
                        df.at[i, 'pack_size'] = "lessequalto5oz"
                    elif lowest_level_pack_value <= 6.0:
                        df.at[i, 'pack_size'] = "lessequalto6oz"
                    elif lowest_level_pack_value <= 8.0:
                        df.at[i, 'pack_size'] = "lessequalto8oz"
                    elif lowest_level_pack_value <= 12.0:
                        df.at[i, 'pack_size'] = "lessequalto12oz"
                    elif lowest_level_pack_value <= 16.0:
                        df.at[i, 'pack_size'] = "lessequalto16oz"
                    elif lowest_level_pack_value <= 20.0:
                        df.at[i, 'pack_size'] = "lessequalto20oz"
                    elif lowest_level_pack_value <= 24.0:
                        df.at[i, 'pack_size'] = "lessequalto24oz"
                    elif lowest_level_pack_value <= 32.0:
                        df.at[i, 'pack_size'] = "lessequalto32oz"
                    elif lowest_level_pack_value > 32.0:
                        df.at[i, 'pack_size'] = "lessequalto100oz"
                    elif lowest_level_pack_value <= 64.0:
                        df.at[i, 'pack_size'] = "lessequalto64oz"
                    elif lowest_level_pack_value <= 100.0:
                        df.at[i, 'pack_size'] = "lessequalto100oz"
                    elif lowest_level_pack_value > 100.0:
                        df.at[i, 'pack_size'] = "greaterthan100oz"
                elif lowest_level_pack_uom == "ea" or lowest_level_pack_uom == "ct" or \
                        lowest_level_pack_uom == "count" or lowest_level_pack_uom == "each":
                    if lowest_level_pack_value <= 1.0:
                        df.at[i, 'pack_size'] = "oneeach"
                    elif lowest_level_pack_value <= 5.0:
                        df.at[i, 'pack_size'] = "lessthan5ea"
                    elif lowest_level_pack_value <= 10.0:
                        df.at[i, 'pack_size'] = "lessthan10ea"
                    elif lowest_level_pack_value <= 15.0:
                        df.at[i, 'pack_size'] = "lessthan15ea"
                    elif lowest_level_pack_value <= 20.0:
                        df.at[i, 'pack_size'] = "lessthan20ea"
                    elif lowest_level_pack_value <= 25.0:
                        df.at[i, 'pack_size'] = "lessthan25ea"
                    elif lowest_level_pack_value <= 30.0:
                        df.at[i, 'pack_size'] = "lessthan30ea"
                    elif lowest_level_pack_value <= 40.0:
                        df.at[i, 'pack_size'] = "lessthan40ea"
                    elif lowest_level_pack_value <= 50.0:
                        df.at[i, 'pack_size'] = "lessthan50ea"
                    elif lowest_level_pack_value <= 75.0:
                        df.at[i, 'pack_size'] = "lessthan75ea"
                    elif lowest_level_pack_value <= 100.0:
                        df.at[i, 'pack_size'] = "lessthan100ea"
                    elif lowest_level_pack_value <= 150.0:
                        df.at[i, 'pack_size'] = "lessthan150ea"
                    elif lowest_level_pack_value <= 200.0:
                        df.at[i, 'pack_size'] = "lessthan200ea"
                    elif lowest_level_pack_value <= 250.0:
                        df.at[i, 'pack_size'] = "lessthan250ea"
                    elif lowest_level_pack_value <= 500.0:
                        df.at[i, 'pack_size'] = "lessthan500ea"
                    elif lowest_level_pack_value <= 1000.0:
                        df.at[i, 'pack_size'] = "lessthan1000ea"
                    elif lowest_level_pack_value > 1000.0:
                        df.at[i, 'pack_size'] = "greaterthan1000ea"
                elif lowest_level_pack_uom == "dz" or lowest_level_pack_uom == "doz" or \
                        lowest_level_pack_uom == "dzn" or lowest_level_pack_uom == "dozen":
                    if lowest_level_pack_value <= 1.0:
                        df.at[i, 'pack_size'] = "TinyDozen"
                    elif lowest_level_pack_value <= 3.0:
                        df.at[i, 'pack_size'] = "LoDozen"
                    elif lowest_level_pack_value <= 6.0:
                        df.at[i, 'pack_size'] = "LoMedDozen"
                    elif lowest_level_pack_value <= 20.0:
                        df.at[i, 'pack_size'] = "MedDozen"
                    elif lowest_level_pack_value > 20.0:
                        df.at[i, 'pack_size'] = "HiDozen"
                elif lowest_level_pack_uom == "ga" or lowest_level_pack_uom == "gll" or \
                        lowest_level_pack_uom == "gal" or lowest_level_pack_uom == "gallon":
                    if lowest_level_pack_value <= 1.0:
                        df.at[i, 'pack_size'] = "TinyGallon"
                    elif lowest_level_pack_value <= 3.0:
                        df.at[i, 'pack_size'] = "LoGallon"
                    elif lowest_level_pack_value <= 5.0:
                        df.at[i, 'pack_size'] = "MedGallon"
                    elif lowest_level_pack_value > 5.0:
                        df.at[i, 'pack_size'] = "HiGallon"
                elif lowest_level_pack_uom == "rl" or lowest_level_pack_uom == "roll":
                    if lowest_level_pack_value <= 10.0:
                        df.at[i, 'pack_size'] = "LoRoll"
                    elif lowest_level_pack_value <= 50.0:
                        df.at[i, 'pack_size'] = "MedRoll"
                    elif lowest_level_pack_value > 50.0:
                        df.at[i, 'pack_size'] = "HiRoll"
                elif lowest_level_pack_uom == "cn" or lowest_level_pack_uom == "can":
                    df.at[i, 'pack_size'] = "Can"
                elif lowest_level_pack_uom == "lt" or lowest_level_pack_uom == "ltr" or \
                        lowest_level_pack_uom == "liter" or lowest_level_pack_uom == "litre":
                    if lowest_level_pack_value <= 1.0:
                        df.at[i, 'pack_size'] = "TinyLiter"
                    elif lowest_level_pack_value <= 4.0:
                        df.at[i, 'pack_size'] = "LoLiter"
                    elif lowest_level_pack_value <= 12.0:
                        df.at[i, 'pack_size'] = "MedLiter"
                    elif lowest_level_pack_value > 12.0:
                        df.at[i, 'pack_size'] = "HiLiter"
                elif lowest_level_pack_uom == "kg" or lowest_level_pack_uom == "kgr" or \
                     lowest_level_pack_uom == "kgm" or lowest_level_pack_uom == "kilo" or \
                     lowest_level_pack_uom == "kilogram":
                    if lowest_level_pack_value <= 3.0:
                        df.at[i, 'pack_size'] = "LoKilogram"
                    elif lowest_level_pack_value <= 12.0:
                        df.at[i, 'pack_size'] = "MedKilogram"
                    elif lowest_level_pack_value > 12.0:
                        df.at[i, 'pack_size'] = "HiKilogram"
                elif lowest_level_pack_uom == "qt" or lowest_level_pack_uom == "qtl" or \
                        lowest_level_pack_uom == "qtd" or lowest_level_pack_uom == "quart":
                    if lowest_level_pack_value <= 4.0:
                        df.at[i, 'pack_size'] = "LoQuart"
                    elif lowest_level_pack_value <= 8.0:
                        df.at[i, 'pack_size'] = "MedQuart"
                    elif lowest_level_pack_value > 8.0:
                        df.at[i, 'pack_size'] = "HiQuart"
                elif lowest_level_pack_uom == "pt" or lowest_level_pack_uom == "ptl" or \
                        lowest_level_pack_uom == "pti" or lowest_level_pack_uom == "pint":
                    if lowest_level_pack_value <= 1.0:
                        df.at[i, 'pack_size'] = "LoPint"
                    elif lowest_level_pack_value <= 4.0:
                        df.at[i, 'pack_size'] = "MedPint"
                    elif lowest_level_pack_value > 4.0:
                        df.at[i, 'pack_size'] = "HiPint"
                elif lowest_level_pack_uom == "kt" or lowest_level_pack_uom == "kit":
                    df.at[i, 'pack_size'] = "Kit"
                elif lowest_level_pack_uom == "ml" or lowest_level_pack_uom == "mlt" or \
                        lowest_level_pack_uom == "milliliter" or lowest_level_pack_uom == "millilitre":
                    if lowest_level_pack_value <= 25.0:
                        df.at[i, 'pack_size'] = "TinyMilliliter"
                    elif lowest_level_pack_value <= 100.0:
                        df.at[i, 'pack_size'] = "LoMilliliter"
                    elif lowest_level_pack_value <= 750.0:
                        df.at[i, 'pack_size'] = "MedMilliliter"
                    elif lowest_level_pack_value > 750.0:
                        df.at[i, 'pack_size'] = "HiMilliliter"
                elif lowest_level_pack_uom == "ft" or lowest_level_pack_uom == "ftq" or \
                     lowest_level_pack_uom == "fot" or lowest_level_pack_uom == "foot" or \
                     lowest_level_pack_uom == "feet":
                    if lowest_level_pack_value <= 5.0:
                        df.at[i, 'pack_size'] = "LoFeet"
                    elif lowest_level_pack_value <= 100.0:
                        df.at[i, 'pack_size'] = "MedFeet"
                    elif lowest_level_pack_value > 100.0:
                        df.at[i, 'pack_size'] = "HiFeet"
                elif lowest_level_pack_uom == "gr" or lowest_level_pack_uom == "grm" or \
                        lowest_level_pack_uom == "gram" or lowest_level_pack_uom == "g":
                    if lowest_level_pack_value <= 5.0:
                        df.at[i, 'pack_size'] = "TinyGram"
                    elif lowest_level_pack_value <= 20.0:
                        df.at[i, 'pack_size'] = "LoGram"
                    elif lowest_level_pack_value <= 100.0:
                        df.at[i, 'pack_size'] = "MedGram"
                    elif lowest_level_pack_value > 100.0:
                        df.at[i, 'pack_size'] = "HiGram"
                elif lowest_level_pack_uom == "st" or lowest_level_pack_uom == "set":
                    df.at[i, 'pack_size'] = "Set"
                elif lowest_level_pack_uom == "bu" or lowest_level_pack_uom == "bushel":
                    df.at[i, 'pack_size'] = "Bushel"
                elif lowest_level_pack_uom == "pk" or lowest_level_pack_uom == "pack":
                    df.at[i, 'pack_size'] = "Pack"
                elif lowest_level_pack_uom == "pr" or lowest_level_pack_uom == "pair":
                    df.at[i, 'pack_size'] = "Pair"
                else:
                    df.at[i, 'pack_size'] = "UnrecognizedPacksize"

                # Flag for multiple levels in the pack size
                if has_two_levels:
                    df.at[i, 'pack_size'] = df.at[i, 'pack_size'] + " withtwolevels"
                if has_three_levels:
                    df.at[i, 'pack_size'] = df.at[i, 'pack_size'] + " withthreelevels"

                has_two_levels = False
                has_three_levels = False

            except:
                df.at[i, 'pack_size'] = "UnrecognizedPacksize"
        except:
            df.at[i, 'pack_size'] = "UnrecognizedPacksize"

    #####
        # Converting storage temperatures to storage descriptors
        #

        # Storage logic for mass create
        if is_usecase_1:
            # Converting the Storage Terms for the Dataset
            for i in range(df.shape[0]):

                try:
                    # Read the storage term provided by the supplier
                    storage = df.loc[i, 'Storage'].lower()
                except:
                    continue

                if storage == "shelf stable" or storage == "ambient" or storage == "dry":
                    desc = df.loc[i, 'description']
                    if "shelfstable" not in desc:
                        df.at[i, 'description'] = desc + " shelfstable"
                elif storage == "frozen" or storage == "fzn" or storage == "freezer" or storage == "frzn" or storage == "freeze":
                    desc = df.loc[i, 'description']
                    if "frozen" not in desc:
                        df.at[i, 'description'] = desc + " frozen"
                elif storage == "refrigerated" or storage == "cooler" or storage == "ref":
                    desc = df.loc[i, 'description']
                    if "ref" not in desc:
                        df.at[i, 'description'] = desc + " ref"

        # Storage logic for product audit
        elif is_usecase_2:
            # Converting the Storage Codes for the Dataset
            for i in range(df.shape[0]):

                try:
                    # Read the product storage code curated by the coordinator
                    storage = df.loc[i, 'Product Storage Code']
                except:
                    continue

                if storage == 18 or storage == 10 or storage == 14 or storage == 16\
                        or storage == 12 or storage == 17 or storage == 15\
                        or storage == 13:
                    desc = df.loc[i, 'description']
                    if "shelfstable" not in desc:
                        df.at[i, 'description'] = desc + " shelfstable"
                elif storage == 50 or storage == 52:
                    desc = df.loc[i, 'description']
                    if "frozen" not in desc:
                        df.at[i, 'description'] = desc + " frozen"
                elif storage == 31 or storage == 35 or storage == 32\
                        or storage == 30 or storage == 36:
                    desc = df.loc[i, 'description']
                    if "ref" not in desc:
                        df.at[i, 'description'] = desc + " ref"
                elif storage == 33:
                    desc = df.loc[i, 'description']
                    if "ref" not in desc:
                        df.at[i, 'description'] = desc + " upper ref"
                    elif "upper" not in desc:
                        df.at[i, 'description'] = desc + " upper"
                elif storage == 34:
                    desc = df.loc[i, 'description']
                    if "ref" not in desc:
                        df.at[i, 'description'] = desc + " upperupper ref"
                    elif "upperupper" not in desc:
                        df.at[i, 'description'] = desc + " upperupper"

        # Storage logic for new item automation
        elif is_usecase_3:
            # Converting the Storage Terms for the Dataset
            for i in range(df.shape[0]):

                try:
                    # Read the storage term provided by the supplier
                    storage = df.loc[i, 'Storage Code'].lower()
                except:
                    continue

                if storage == "dry food" or storage == "dry non food":
                    desc = df.loc[i, 'description']
                    if "shelfstable" not in desc:
                        df.at[i, 'description'] = desc + " shelfstable"
                elif storage == "frozen":
                    desc = df.loc[i, 'description']
                    if "frozen" not in desc:
                        df.at[i, 'description'] = desc + " frozen"
                elif storage == "ref":
                    desc = df.loc[i, 'description']
                    if "ref" not in desc:
                        df.at[i, 'description'] = desc + " ref"

                # set the value for food_nonfood
                if storage == "dry non food":
                    df.at[i, 'food_nonfood'] = "nonfood"
                else:
                    df.at[i, 'food_nonfood'] = "food"

        # Storage logic for general file input
        else:
            # Converting the Storage Temps for the Dataset
            for i in range(df.shape[0]):

                min_temp = df.loc[i, 'temp_min']
                max_temp = df.loc[i, 'temp_max']

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
                    desc = df.loc[i, 'description']
                    if "frozen" not in desc:
                        df.at[i, 'description'] = desc + " frozen"
                elif int(max_temp) < 28:
                    desc = df.loc[i, 'description']
                    if "frozen" not in desc:
                        df.at[i, 'description'] = desc + " frozen"
                # Logic for ref storage
                elif int(max_temp) <= 45:
                    desc = df.loc[i, 'description']
                    if "ref" not in desc:
                        df.at[i, 'description'] = desc + " ref"
                # Logic for upper ref storage
                elif int(max_temp) <= 55:
                    desc = df.loc[i, 'description']
                    if "ref" not in desc:
                        df.at[i, 'description'] = desc + " upper ref"
                    elif "upper" not in desc:
                        df.at[i, 'description'] = desc + " upper"
                # Logic for upperupper ref storage
                elif int(max_temp) <= 70:
                    desc = df.loc[i, 'description']
                    if "ref" not in desc:
                        df.at[i, 'description'] = desc + " upperupper ref"
                    elif "upperupper" not in desc:
                        df.at[i, 'description'] = desc + " upperupper"
                # Logic for shelfstable storage
                elif int(max_temp) >= 71:
                    desc = df.loc[i, 'description']
                    if "shelfstable" not in desc:
                        df.at[i, 'description'] = desc + " shelfstable"

    #####
    # Concatenating the description, coo, and pack_size into a single text field
    #

    # Concatenating description-related fields for the Training Dataset
    if df.coo is not None:
        df['description'] = df.description.astype(str).str.cat(df.coo.astype(str), sep=' ')
    if df.pack_size is not None:
        df['description'] = df.description.astype(str).str.cat(df.pack_size.astype(str), sep=' ')
    if df.brand is not None:
        df['description'] = df.description.astype(str).str.cat(df.brand.astype(str), sep=' ')
    if df.manufacturer is not None:
        df['description'] = df.description.astype(str).str.cat(df.manufacturer.astype(str), sep=' ')

    # Prune the description one last time to make sure it fits within the max sequence length
    df = prune_data_final(df)

    # Perform a final cleanup to manage None values
    df['description'] = df['description'].apply(clean_none)

    return df, is_usecase_1, is_usecase_2, is_usecase_3


def convert_packsize(packsize_text):

    #####
    # Converting pack sizes to general terms more easily managed by the model
    #

    packsize = packsize_text

    time.sleep(.3)

    # Check to see if uom is "can"
    if "can" in packsize:
        packsize = "Can"

        return packsize

    try:
        int(packsize)
        return packsize
    except:
        packsize = str(packsize)

    try:

        has_two_levels = False
        has_three_levels = False

        # Parse the pack size
        slash_split_pack = packsize.split('/')  # parses by '/' and returns a list of the substrings

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
                    packsize = "lessequalto1lb"
                elif lowest_level_pack_value <= 5.0:
                    packsize = "lessequalto5lb"
                elif lowest_level_pack_value <= 10.0:
                    packsize = "lessequalto10lb"
                elif lowest_level_pack_value <= 20.0:
                    packsize = "lessequalto20lb"
                elif lowest_level_pack_value <= 100.0:
                    packsize = "lessequalto100lb"
                elif lowest_level_pack_value > 100.0:
                    packsize = "greaterthan100lb"
            elif lowest_level_pack_uom == "oz" or lowest_level_pack_uom == "oza" or \
                    lowest_level_pack_uom == "onz" or lowest_level_pack_uom == "ounce":
                if lowest_level_pack_value <= 1.0:
                    packsize = "lessequalto1oz"
                elif lowest_level_pack_value <= 2.0:
                    packsize = "lessequalto2oz"
                elif lowest_level_pack_value <= 3.0:
                    packsize = "lessequalto3oz"
                elif lowest_level_pack_value <= 4.0:
                    packsize = "lessequalto4oz"
                elif lowest_level_pack_value <= 5.0:
                    packsize = "lessequalto5oz"
                elif lowest_level_pack_value <= 6.0:
                    packsize = "lessequalto6oz"
                elif lowest_level_pack_value <= 8.0:
                    packsize = "lessequalto8oz"
                elif lowest_level_pack_value <= 12.0:
                    packsize = "lessequalto12oz"
                elif lowest_level_pack_value <= 16.0:
                    packsize = "lessequalto16oz"
                elif lowest_level_pack_value <= 20.0:
                    packsize = "lessequalto20oz"
                elif lowest_level_pack_value <= 24.0:
                    packsize = "lessequalto24oz"
                elif lowest_level_pack_value <= 32.0:
                    packsize = "lessequalto32oz"
                elif lowest_level_pack_value > 32.0:
                    packsize = "lessequalto100oz"
                elif lowest_level_pack_value <= 64.0:
                    packsize = "lessequalto64oz"
                elif lowest_level_pack_value <= 100.0:
                    packsize = "lessequalto100oz"
                elif lowest_level_pack_value > 100.0:
                    packsize = "greaterthan100oz"
            elif lowest_level_pack_uom == "ea" or lowest_level_pack_uom == "ct" or \
                    lowest_level_pack_uom == "count" or lowest_level_pack_uom == "each":
                if lowest_level_pack_value <= 1.0:
                    packsize = "oneeach"
                elif lowest_level_pack_value <= 10.0:
                    packsize = "lessthan10ea"
                elif lowest_level_pack_value <= 20.0:
                    packsize = "lessthan20ea"
                elif lowest_level_pack_value <= 50.0:
                    packsize = "lessthan50ea"
                elif lowest_level_pack_value <= 100.0:
                    packsize = "lessthan100ea"
                elif lowest_level_pack_value <= 250.0:
                    packsize = "lessthan250ea"
                elif lowest_level_pack_value <= 500.0:
                    packsize = "lessthan500ea"
                elif lowest_level_pack_value <= 1000.0:
                    packsize = "lessthan1000ea"
                elif lowest_level_pack_value > 1000.0:
                    packsize = "greaterthan1000ea"
            elif lowest_level_pack_uom == "dz" or lowest_level_pack_uom == "doz" or \
                    lowest_level_pack_uom == "dzn" or lowest_level_pack_uom == "dozen":
                if lowest_level_pack_value <= 1.0:
                    packsize = "TinyDozen"
                elif lowest_level_pack_value <= 6.0:
                    packsize = "LoDozen"
                elif lowest_level_pack_value <= 20.0:
                    packsize = "MedDozen"
                elif lowest_level_pack_value > 20.0:
                    packsize = "HiDozen"
            elif lowest_level_pack_uom == "ga" or lowest_level_pack_uom == "gll" or \
                    lowest_level_pack_uom == "gal" or lowest_level_pack_uom == "gallon":
                if lowest_level_pack_value <= 1.0:
                    packsize = "TinyGallon"
                elif lowest_level_pack_value <= 3.0:
                    packsize = "LoGallon"
                elif lowest_level_pack_value <= 5.0:
                    packsize = "MedGallon"
                elif lowest_level_pack_value > 5.0:
                    packsize = "HiGallon"
            elif lowest_level_pack_uom == "rl" or lowest_level_pack_uom == "roll":
                if lowest_level_pack_value <= 10.0:
                    packsize = "LoRoll"
                elif lowest_level_pack_value <= 50.0:
                    packsize = "MedRoll"
                elif lowest_level_pack_value > 50.0:
                    packsize = "HiRoll"
            elif lowest_level_pack_uom == "cn" or lowest_level_pack_uom == "can":
                packsize = "Can"
            elif lowest_level_pack_uom == "lt" or lowest_level_pack_uom == "ltr" or \
                    lowest_level_pack_uom == "liter" or lowest_level_pack_uom == "litre":
                if lowest_level_pack_value <= 1.0:
                    packsize = "TinyLiter"
                elif lowest_level_pack_value <= 4.0:
                    packsize = "LoLiter"
                elif lowest_level_pack_value <= 12.0:
                    packsize = "MedLiter"
                elif lowest_level_pack_value > 12.0:
                    packsize = "HiLiter"
            elif lowest_level_pack_uom == "kg" or lowest_level_pack_uom == "kgr" or \
                 lowest_level_pack_uom == "kgm" or lowest_level_pack_uom == "kilo" or \
                 lowest_level_pack_uom == "kilogram":
                if lowest_level_pack_value <= 3.0:
                    packsize = "LoKilogram"
                elif lowest_level_pack_value <= 12.0:
                    packsize = "MedKilogram"
                elif lowest_level_pack_value > 12.0:
                    packsize = "HiKilogram"
            elif lowest_level_pack_uom == "qt" or lowest_level_pack_uom == "qtl" or \
                    lowest_level_pack_uom == "qtd" or lowest_level_pack_uom == "quart":
                if lowest_level_pack_value <= 4.0:
                    packsize = "LoQuart"
                elif lowest_level_pack_value <= 8.0:
                    packsize = "MedQuart"
                elif lowest_level_pack_value > 8.0:
                    packsize = "HiQuart"
            elif lowest_level_pack_uom == "pt" or lowest_level_pack_uom == "ptl" or \
                    lowest_level_pack_uom == "pti" or lowest_level_pack_uom == "pint":
                if lowest_level_pack_value <= 1.0:
                    packsize = "LoPint"
                elif lowest_level_pack_value <= 4.0:
                    packsize = "MedPint"
                elif lowest_level_pack_value > 4.0:
                    packsize = "HiPint"
            elif lowest_level_pack_uom == "kt" or lowest_level_pack_uom == "kit":
                packsize = "Kit"
            elif lowest_level_pack_uom == "ml" or lowest_level_pack_uom == "mlt" or \
                    lowest_level_pack_uom == "milliliter" or lowest_level_pack_uom == "millilitre":
                if lowest_level_pack_value <= 25.0:
                    packsize = "TinyMilliliter"
                elif lowest_level_pack_value <= 100.0:
                    packsize = "LoMilliliter"
                elif lowest_level_pack_value <= 750.0:
                    packsize = "MedMilliliter"
                elif lowest_level_pack_value > 750.0:
                    packsize = "HiMilliliter"
            elif lowest_level_pack_uom == "ft" or lowest_level_pack_uom == "ftq" or \
                 lowest_level_pack_uom == "fot" or lowest_level_pack_uom == "foot" or \
                 lowest_level_pack_uom == "feet":
                if lowest_level_pack_value <= 5.0:
                    packsize = "LoFeet"
                elif lowest_level_pack_value <= 100.0:
                    packsize = "MedFeet"
                elif lowest_level_pack_value > 100.0:
                    packsize = "HiFeet"
            elif lowest_level_pack_uom == "gr" or lowest_level_pack_uom == "grm" or \
                    lowest_level_pack_uom == "gram":
                if lowest_level_pack_value <= 5.0:
                    packsize = "TinyGram"
                elif lowest_level_pack_value <= 20.0:
                    packsize = "LoGram"
                elif lowest_level_pack_value <= 100.0:
                    packsize = "MedGram"
                elif lowest_level_pack_value > 100.0:
                    packsize = "HiGram"
            elif lowest_level_pack_uom == "st" or lowest_level_pack_uom == "set":
                packsize = "Set"
            elif lowest_level_pack_uom == "bu" or lowest_level_pack_uom == "bushel":
                packsize = "Bushel"
            elif lowest_level_pack_uom == "pk" or lowest_level_pack_uom == "pack":
                packsize = "Pack"
            elif lowest_level_pack_uom == "pr" or lowest_level_pack_uom == "pair":
                packsize = "Pair"

            # Flag for multiple levels in the pack size
            if has_two_levels:
                packsize = packsize + " withtwolevels"
            if has_three_levels:
                packsize = packsize + " withthreelevels"

            has_two_levels = False
            has_three_levels = False

            return packsize

        except:
            # print("Exception encountered when attempting to convert Pack Size.")
            packsize = "UnrecognizedPacksize"

    except:
        # print("Exception encountered when attempting to parse the Pack Size.")
        packsize = "UnrecognizedPacksize"
