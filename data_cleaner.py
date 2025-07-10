"""
DATA CLEANER

Developed By: Thomas Vaughn
Version: 1.1.4
Last Update Date: 7/10/2025

This script is designed to be imported as a module from the Data Preprocessor
and Product Classifier scripts.

At a high-level, the Data Cleaner is designed to perform basic cleaning
tasks which support the larger preprocessing effort in preparation
for model training or prediction.

This script contains various functions for cleaning data for various
features for Product Classification.  Each function is designed
to meet specific business needs related to the individual features.
The script also contains a couple of functions which tie the
specialized functions together for ease of use when called from
the preprocessor or predictor scripts.

This script contains the following functions:

    * clean_special_characters - performs a simple ascii encoding/decoding
    * clean_none - cleans none and nan values along with extra spaces
    * prune_desc - prunes the description to 16 terms max
    * prune_mfr_brand - prunes the manufacturer and brand to 3 terms max
    * prune_coo - prunes the country or origin to a single term
    * prune_final_desc - prunes the description after additional values have been added
    * prune_data_final - a helper function to send data through the prune_final_desc function
    * clean_gtin - if value cannot be converted to an integer, returns a zero
    * clean_coo - cleans the country of origin
    * clean_description - cleans and lemmatizes the description
    * clean_packsize - cleans the packsize
    * clean_mfr_brand - cleans and lemmatizes the manufacturer and brand
    * convert_to_fah - converts Celsius temps to Fahrenheit
    * clean_data - calls appropriate functions to clean the primary features in the product data
    * clean_first_pass - calls the clean_special_characters function for multiple features

"""

import cleantext
import numpy
import pandas as pd
import time
from nltk.stem import WordNetLemmatizer

pd.options.mode.chained_assignment = None  # Disable chained assignment warnings

wnl = WordNetLemmatizer()


def clean_special_characters(text):

    text = str(text)

    text = text.encode('ascii', 'ignore')
    text = text.decode()

    return text


def clean_none(text):

    text = str(text)

    while 'none' in text:
        text = text.replace('none', '')
    while 'nan' in text:
        text = text.replace('nan', '')
    while "      " in text:
        text = text.replace("      ", " ")
    while "     " in text:
        text = text.replace("     ", " ")
    while "    " in text:
        text = text.replace("    ", " ")
    while "   " in text:
        text = text.replace("   ", " ")
    while "  " in text:
        text = text.replace("  ", " ")

    return text


def prune_desc(text):

    text = str(text)
    terms = text.split()

    while len(terms) > 16:
        terms.pop(len(terms) - 1)

    pruned_desc = ""

    for i in range(len(terms)):
        if i == 0:
            pruned_desc = terms[i]
        else:
            pruned_desc = pruned_desc + " " + terms[i]

    return pruned_desc


def prune_mfr_brand(text):

    text = str(text)
    terms = text.split()

    if text == "nan":
        return

    while len(terms) > 3:
        terms.pop(len(terms) - 1)

    pruned_text = ""

    for i in range(len(terms)):
        if i == 0:
            pruned_text = terms[i]
        else:
            pruned_text = pruned_text + " " + terms[i]

    return pruned_text


def prune_coo(text):

    text = str(text)
    terms = text.split()

    if terms[0] == "" and terms[1] != "":
        pruned_coo = terms[0]
    elif terms[0] != "":
        pruned_coo = terms[0]
    else:
        pruned_coo = ""

    return pruned_coo


def prune_final_desc(text):

    text = str(text)
    terms = text.split()

    while len(terms) > 26:
        terms.pop(len(terms) - 1)

    pruned_desc = ""

    for i in range(len(terms)):
        if i == 0:
            pruned_desc = terms[i]
        else:
            pruned_desc = pruned_desc + " " + terms[i]

    return pruned_desc


def prune_data_final(data):

    data['description'] = data['description'].apply(prune_final_desc)

    return data


def clean_gtin(text):

    original_text = text

    try:

        test_text = int(text)
    except:
        original_text = "0"

    return original_text


def clean_coo(text):

    text = str(text)

    text = cleantext.clean(
        text,
        clean_all=False,
        extra_spaces=True,
        stemming=False,
        stopwords=False,
        lowercase=True,
        numbers=True,
        punct=True,
        stp_lang='english'
    )

    text = text.encode('ascii', 'ignore')
    text = text.decode()

    return text


def clean_description(text):

    text = str(text)

    while '1%' in text:
        text = text.replace('1%', 'onepercent')
    while '2%' in text:
        text = text.replace('2%', 'twopercent')
    while '1000' in text:
        text = text.replace('1000', ' thousand ')
    while '100' in text:
        text = text.replace('100', ' hundred ')
    while '10' in text:
        text = text.replace('10', ' ten ')
    while ' S/S' in text:
        text = text.replace(' S/S', ' SSLASHS')
    while ' F/' in text:
        text = text.replace(' F/', ' FSLASH')
    while ' W/' in text:
        text = text.replace(' W/', ' WSLASH')
    while ' W/O' in text:
        text = text.replace(' W/O', ' WSLASHO')
    while ' 1130A' in text:
        text = text.replace(' 1130A', ' PORTIONED')
    while ' 1130' in text:
        text = text.replace(' 1130', ' PORTIONED')
    while ' 1123D' in text:
        text = text.replace(' 1123D', ' PORTIONED')
    while ' 1123A' in text:
        text = text.replace(' 1123A', ' PORTIONED')
    while ' 1123' in text:
        text = text.replace(' 1123', ' PORTIONED')
    while '""""' in text:
        text = text.replace('""""', '"')
    while '"""' in text:
        text = text.replace('"""', '"')
    while '""' in text:
        text = text.replace('""', '"')
    while '#' in text:
        text = text.replace('#', ' poundsign ')
    while ':' in text:
        text = text.replace(':', ' colonpunctuation ')
    while '"' in text:
        text = text.replace('"', ' quotationmark ')
    while '+' in text:
        text = text.replace('+', ' plussign ')
    while "'" in text:
        text = text.replace("'", ' apostrophe ')
    while '%' in text:
        text = text.replace('%', ' percent ')

    text = cleantext.clean(
        text,
        clean_all=False,
        extra_spaces=True,
        stemming=False,
        stopwords=False,
        lowercase=True,
        numbers=True,
        punct=False,
        stp_lang='english'
    )

    text = text.encode('ascii', 'ignore')
    text = text.decode()

    while ' the ' in text:
        text = text.replace(' the ', ' ')
    while 'shelf stable' in text:
        text = text.replace('shelf stable', 'shelfstable')
    while 'raised-w/o-antibiotics' in text:
        text = text.replace('raised-w/o-antibiotics', ' raisedwoantibiotics ')
    while 'vacuum-pack' in text:
        text = text.replace('vacuum-pack', ' vacuumpack ')
    while 'vacuum pack' in text:
        text = text.replace('vacuum pack', ' vacuumpack ')
    while 't-bone' in text:
        text = text.replace('t-bone', ' tbone ')
    while ' a ' in text:
        text = text.replace(' a ', ' ')
    while ' b ' in text:
        text = text.replace(' b ', ' ')
    while ' c ' in text:
        text = text.replace(' c ', ' ')
    while ' d ' in text:
        text = text.replace(' d ', ' ')
    while ' e ' in text:
        text = text.replace(' e ', ' ')
    while ' and ' in text:
        text = text.replace(' and ', ' ')
    while ' with ' in text:
        text = text.replace(' with ', ' ')
    while '-' in text:
        text = text.replace("-", " ")
    while "\\\\\\\\" in text:
        text = text.replace("\\\\\\\\", " ")
    while "\\\\" in text:
        text = text.replace("\\\\", " ")
    while "\n" in text:
        text = text.replace("\n", " ")
    while "." in text:
        text = text.replace(".", " ")
    while ",,," in text:
        text = text.replace(",,,", ",")
    while ",," in text:
        text = text.replace(",,", ",")
    while "," in text:
        text = text.replace(",", " ")
    while "!" in text:
        text = text.replace("!", " ")
    while "*" in text:
        text = text.replace("*", " ")
    while ";" in text:
        text = text.replace(";", " ")
    while "?" in text:
        text = text.replace("?", " ")
    while "_" in text:
        text = text.replace("_", " ")
    while "~" in text:
        text = text.replace("~", " ")
    while "/" in text:
        text = text.replace("/", " ")
    while "&" in text:
        text = text.replace("&", " ")
    while "''" in text:
        text = text.replace("''", " ")
    while "/" in text:
        text = text.replace("/", " ")
    while "(" in text:
        text = text.replace("(", " ")
    while ")" in text:
        text = text.replace(")", " ")
    while "``" in text:
        text = text.replace("``", "")
    while "''" in text:
        text = text.replace("''", "")
    while "      " in text:
        text = text.replace("      ", " ")
    while "    " in text:
        text = text.replace("    ", " ")
    while "  " in text:
        text = text.replace("  ", " ")

    # lemmatize the text
    lemmatized_text = ""
    words_in_text = text.split()
    for i in range(len(words_in_text)):
        word = words_in_text[i]
        if word == "cookies":  # wnl lemmatizer converts cookies to cooky so this is a workaround
            word = "cookie"
        elif word == "hoagies":  # wnl lemmatizer converts hoagies to hoagy so this is a workaround
            word = "hoagie"
        else:
            word = wnl.lemmatize(word)
        if i == 0:
            lemmatized_text = word
        else:
            lemmatized_text = lemmatized_text + " " + str(word)

    return lemmatized_text


def clean_packsize(text):

    text = str(text)

    while "1N" in text:
        text = text.replace("1N", "ea")
    while "H87" in text:
        text = text.replace("H87", "ea")
    while "1n" in text:
        text = text.replace("1n", "ea")
    while "h87" in text:
        text = text.replace("h87", "ea")
    while "15/1 DZ" in text:
        text = text.replace("15/1 DZ", "15 DZ")
    while "30/1 DZ" in text:
        text = text.replace("30/1 DZ", "30 DZ")

    text = cleantext.clean(
        text,
        clean_all=False,
        extra_spaces=True,
        stemming=False,
        stopwords=False,
        lowercase=True,
        numbers=False,
        punct=False,
        stp_lang='english'
    )

    text = text.encode('ascii', 'ignore')
    text = text.decode()

    while "\\\\\\\\" in text:
        text = text.replace("\\\\\\\\", " ")
    while "\\\\" in text:
        text = text.replace("\\\\", " ")
    while "\n" in text:
        text = text.replace("\n", " ")
    while ",,," in text:
        text = text.replace(",,,", ",")
    while ",," in text:
        text = text.replace(",,", ",")
    while "," in text:
        text = text.replace(",", " ")
    while "!" in text:
        text = text.replace("!", " ")
    while "*" in text:
        text = text.replace("*", " ")
    while ";" in text:
        text = text.replace(";", "/")
    while "?" in text:
        text = text.replace("?", " ")
    while "_" in text:
        text = text.replace("_", " ")
    while "~" in text:
        text = text.replace("~", " ")
    while "&" in text:
        text = text.replace("&", " ")
    while "''" in text:
        text = text.replace("''", " ")
    while "(" in text:
        text = text.replace("(", " ")
    while ")" in text:
        text = text.replace(")", " ")
    while "``" in text:
        text = text.replace("``", "")
    while "''" in text:
        text = text.replace("''", "")
    while '""""' in text:
        text = text.replace('""""', '"')
    while '"""' in text:
        text = text.replace('"""', '"')
    while '""' in text:
        text = text.replace('""', '"')
    while 'cn' in text:
        text = text.replace('cn', 'can')
    while '#a' in text:
        text = text.replace('#a', ' lba')
    while 'X' in text:
        text = text.replace('X', '/')
    while 'x' in text:
        text = text.replace('x', '/')
    while ' / ' in text:
        text = text.replace(' / ', ' /')
    while '/ ' in text:
        text = text.replace('/ ', ' /')
    while ' /' in text:
        text = text.replace(' /', '/')
    while "       " in text:
        text = text.replace("       ", " ")
    while "      " in text:
        text = text.replace("      ", " ")
    while "     " in text:
        text = text.replace("     ", " ")
    while "    " in text:
        text = text.replace("    ", " ")
    while "   " in text:
        text = text.replace("   ", " ")
    while "  " in text:
        text = text.replace("  ", " ")

    return text


def clean_mfr_brand(text):

    text = str(text)

    while '1%' in text:
        text = text.replace('1%', 'onepercent')
    while '2%' in text:
        text = text.replace('2%', 'twopercent')
    while '1000' in text:
        text = text.replace('1000', ' thousand ')
    while '100' in text:
        text = text.replace('100', ' hundred ')
    while '10' in text:
        text = text.replace('10', ' ten ')

    text = cleantext.clean(
        text,
        clean_all=False,
        extra_spaces=True,
        stemming=False,
        stopwords=False,
        lowercase=True,
        numbers=True,
        punct=False,
        stp_lang='english'
    )

    text = text.encode('ascii', 'ignore')
    text = text.decode()

    while ' a ' in text:
        text = text.replace(' a ', ' ')
    while ' and ' in text:
        text = text.replace(' and ', ' ')
    while ' the ' in text:
        text = text.replace(' the ', ' ')
    while '-' in text:
        text = text.replace("-", " ")
    while "\\\\\\\\" in text:
        text = text.replace("\\\\\\\\", " ")
    while "\\\\" in text:
        text = text.replace("\\\\", " ")
    while "\n" in text:
        text = text.replace("\n", " ")
    while "." in text:
        text = text.replace(".", " ")
    while ",,," in text:
        text = text.replace(",,,", ",")
    while ",," in text:
        text = text.replace(",,", ",")
    while "," in text:
        text = text.replace(",", " ")
    while "!" in text:
        text = text.replace("!", " ")
    while "*" in text:
        text = text.replace("*", " ")
    while ";" in text:
        text = text.replace(";", " ")
    while "?" in text:
        text = text.replace("?", " ")
    while "_" in text:
        text = text.replace("_", " ")
    while "~" in text:
        text = text.replace("~", " ")
    while "/" in text:
        text = text.replace("/", " ")
    while "&" in text:
        text = text.replace("&", " ")
    while "''" in text:
        text = text.replace("''", " ")
    while "/" in text:
        text = text.replace("/", " ")
    while "(" in text:
        text = text.replace("(", " ")
    while ")" in text:
        text = text.replace(")", " ")
    while "``" in text:
        text = text.replace("``", "")
    while "''" in text:
        text = text.replace("''", "")
    while '""""' in text:
        text = text.replace('""""', '"')
    while '"""' in text:
        text = text.replace('"""', '"')
    while '""' in text:
        text = text.replace('""', '"')
    while '#' in text:
        text = text.replace('#', ' poundsign ')
    while ':' in text:
        text = text.replace(':', ' colonpunctuation ')
    while '"' in text:
        text = text.replace('"', ' quotationmark ')
    while '+' in text:
        text = text.replace('+', ' plussign ')
    while "'" in text:
        text = text.replace("'", ' apostrophe ')
    while '%' in text:
        text = text.replace('%', ' percent ')
    while "      " in text:
        text = text.replace("      ", " ")
    while "    " in text:
        text = text.replace("    ", " ")
    while "  " in text:
        text = text.replace("  ", " ")

    # lemmatize the text
    lemmatized_text = ""
    words_in_text = text.split()
    for i in range(len(words_in_text)):
        word = words_in_text[i]
        if word == "cookies":  # wnl lemmatizer converts cookies to cooky so this is a workaround
            word = "cookie"
        elif word == "hoagies":  # wnl lemmatizer converts hoagies to hoagy so this is a workaround
            word = "hoagie"
        else:
            word = wnl.lemmatize(word)
        if i == 0:
            lemmatized_text = word
        else:
            lemmatized_text = lemmatized_text + " " + str(word)

    return lemmatized_text


def convert_to_fah(source_temp):

    temp = source_temp
    temp = (temp * 1.8) + 32
    temp = round(temp)

    return temp


def clean_data(source_text):

    data = source_text

    # Convert celsius values to fahrenheit
    try:
        print("Cleaning the storage temperature data...")

        time.sleep(.3)

        for i in range(len(data.index.tolist())):
            if (data.loc[i, 'temp_min_uom'] == 'CEL') or (data.loc[i, 'temp_min_uom'] == 'CE'):
                data.loc[i, 'temp_min'] = convert_to_fah(data.loc[i, 'temp_min'])
            if (data.loc[i, 'temp_max_uom'] == 'CEL') or (data.loc[i, 'temp_max_uom'] == 'CE'):
                data.loc[i, 'temp_max'] = convert_to_fah(data.loc[i, 'temp_max'])
            temp_min = data['temp_min'].loc[i]
            if isinstance(temp_min, numpy.float64):
                temp_min = temp_min.item()
            if isinstance(temp_min, str):
                temp_min = int(temp_min)
            data.loc[i, 'temp_min'] = temp_min
            temp_min_uom = data['temp_min_uom'].loc[i]
            if isinstance(temp_min_uom, numpy.float64):
                temp_min_uom = temp_min_uom.item()
            temp_min_uom = str(temp_min_uom)
            data.loc[i, 'temp_min_uom'] = temp_min_uom
            temp_max = data['temp_max'].loc[i]
            if isinstance(temp_max, numpy.float64):
                temp_max = temp_max.item()
            if isinstance(temp_max, str):
                temp_max = int(temp_max)
            data.loc[i, 'temp_max'] = temp_max
            temp_max_uom = data['temp_max_uom'].loc[i]
            if isinstance(temp_max_uom, numpy.float64):
                temp_max_uom = temp_max_uom.item()
            temp_max_uom = str(temp_max_uom)
            data.loc[i, 'temp_max_uom'] = temp_max_uom
        print("Finished cleaning the storage temperature data.\n")
    except Exception as e:
        print(e)
        print("Unable to locate required temperature data.\n")

    # Clean the text data - one column at a time
    try:
        print("Cleaning the gtin data...")
        data['gtin'] = data['gtin'].apply(clean_gtin)
        print("Finished cleaning the gtin data.\n")
    except:
        print("No gtin data found.\n")
    try:
        print("Cleaning the description data...")
        data['description'] = data['description'].apply(clean_description)
        data['description'] = data['description'].apply(prune_desc)
        print("Finished cleaning the description data.\n")
    except:
        print("No description data found.\n")
    try:
        print("Cleaning the country of origin data...")
        data['coo'] = data['coo'].apply(clean_coo)
        data['coo'] = data['coo'].apply(prune_coo)
        print("Finished cleaning the country of origin data.\n")
    except:
        print("No gpc data found.\n")
    try:
        print("Cleaning the manufacturer data...")
        data['manufacturer'] = data['manufacturer'].apply(clean_mfr_brand)
        data['manufacturer'] = data['manufacturer'].apply(prune_mfr_brand)
        print("Finished cleaning the manufacturer data.\n")
    except:
        print("No manufacturer data found.\n")
    try:
        print("Cleaning the brand data...")
        data['brand'] = data['brand'].apply(clean_mfr_brand)
        data['brand'] = data['brand'].apply(prune_mfr_brand)
        print("Finished cleaning the brand data.\n")
    except:
        print("No brand data found.\n")
    try:
        print("Cleaning the pack size data...")
        data['pack_size'] = data['pack_size'].apply(clean_packsize)
        print("Finished cleaning the pack size data.\n")
    except:
        print("No pack_size data found.\n")

    return data


def clean_first_pass(source_text):

    data = source_text

    # Clean the text data - one column at a time
    try:
        print("Cleaning the description data...")
        data['description'] = data['description'].apply(clean_special_characters)
        print("Finished cleaning the description data.\n")
    except:
        print("No description data found.\n")
    try:
        print("Cleaning the country of origin data...")
        data['coo'] = data['coo'].apply(clean_special_characters)
        print("Finished cleaning the country of origin data.\n")
    except:
        print("No COO data found.\n")
    try:
        print("Cleaning the manufacturer data...")
        data['manufacturer'] = data['manufacturer'].apply(clean_special_characters)
        print("Finished cleaning the manufacturer data.\n")
    except:
        print("No manufacturer data found.\n")
    try:
        print("Cleaning the brand data...")
        data['brand'] = data['brand'].apply(clean_special_characters)
        print("Finished cleaning the brand data.\n")
    except:
        print("No brand data found.\n")
    try:
        print("Cleaning the pack size data...")
        data['pack_size'] = data['pack_size'].apply(clean_special_characters)
        print("Finished cleaning the pack size data.\n")
    except:
        print("No pack_size data found.\n")

    return data
