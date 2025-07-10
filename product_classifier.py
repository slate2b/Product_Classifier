"""
PRODUCT CLASSIFIER

Developed By: Thomas Vaughn
Version: 1.1.4
Last Update Date: 7/10/2025

The models used in this application are built using a custom architecture which utilizes an Embedding layer, a
Positional Encoding layer, a Multihead Attention layer, Dual Multilayer Encoders, and a final Classification layer.

Each of the Dual Encoders follows design elements similar to Encoder-Only Transformer models like BERT.  Although the
encoders are probably the most notable components in architecture, it is the collaboration between the Embedding,
Positional Encoding, Multihead Attention, and the Dual Encoders which makes the classification task successful.

TEXT PREPROCESSING

The Product Classifier models are designed for text-based input, but unlike many popular models which are built to
make sense of natural language (statements, questions, common forms of human speech), these models are designed to make
sense of product data in as it is commonly found on a product specification sheet.  This includes the description, brand
name, manufacturer name, pack size, country of origin, and storage data.  Like other text-based models, the Product
Classifier models require the text to be preprocessed into forms that help them function effectively.  The classifier
utilizes a data cleaner and a preprocessor which are custom designed for these models.  Once the text is preprocessed,
the classifier program tokenizes the text (converts each text term to a numeric value) which is the last step before
feeding the data to the models.

MODEL PROCESSING: HIGH-LEVEL EXPLANATION

The Product Classifier models used in this classifier start by passing the input data through an Embedding Layer.
This embedding layer takes the data and represents each tokenized term as an n-size tensor.  In other words, it takes
the data represents each term (for example: the word "pizza") as a collection of many numbers.  The embedding layer's
job is to find the most effective collection of numbers to represent each term.  It improves over time during the
training process.

The data is then sent to the Positional Encoding layer.  The positional encoding layer takes the data from the embedding
layer and modifies the numeric representation taking into account the position of each term with the sequence.  For
example, if one of the input records is "Pizza Pepperoni Frozen", then "Pizza" is in the first position, "Pepperoni" is
in the second position, and "Frozen" is in the third position.  The positional encoding modifies the representations of
each term to draw attention to the fact that "Pizza" is not only part of the data, but that it was the first term in the
data.  This means that the same token (term) would be represented by a different tensor (collection of numbers) if it
were in position 1 than if it were in position 2.

At this point, the data is sent to the two encoders in this dual-encoder architecture.  Each encoder includes multiple
encoding layers, and each encoding layer feeds data through its own Multihead Attention module and Positionwise Feed
Forward module.  The Multihead Attention layer takes the data and sends it through 3 different linear layers to
represent the Query, Key, and Value views following the self-attention approach presented in the famous paper
"Attention Is All You Need" from the researchers at Google.  This allows the models to focus on the values of a term
that are most significant, therefore improving the models' ability to properly interpret the input.  Once the data
passes through the Multihead Attention module, it is then sent through a Positionwise Feed Forward module composed of
2 linear layers.  So in each encoder, the data passes through multiple layers of multihead attention and feed forward
processing, formulating a better understanding of data with each layer.

Finally, the output from the two encoders is sent to a linear classification layer which makes the final decision
about which categories are the best fit for the given data.

"""


import csv
import os
import sys
import torch
from data_preprocessor import prepare_dataset, convert_packsize
from data_cleaner import clean_description, clean_mfr_brand, clean_packsize, \
     prune_desc, prune_mfr_brand, clean_special_characters
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch import nn
import torchtext.vocab
import ctypes
import math
from datetime import datetime
from torchtext.data.functional import to_map_style_dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Progressbar
import threading
from nltk.stem import WordNetLemmatizer
import traceback

ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 6)

wnl = WordNetLemmatizer()

CONFIDENCE_MIN_1 = 0.33
CONFIDENCE_MIN_2 = 0.03
CONFIDENCE_MIN_3 = 0.03

MODEL_1_PATH = './Models/product_classifier_1.pt'
VOCAB_1_PATH = './Models/product_classifier_1_vocab.pt'
MODEL_1_MAX_SEQ_LEN = 26
MODEL_1_CONFIDENCE_MODIFIER = 1.0
MODEL_2_PATH = './Models/product_classifier_2.pt'
VOCAB_2_PATH = './Models/product_classifier_2_vocab.pt'
MODEL_2_MAX_SEQ_LEN = 26
MODEL_2_CONFIDENCE_MODIFIER = 1.0
MODEL_3_PATH = './Models/product_classifier_3.pt'
VOCAB_3_PATH = './Models/product_classifier_3_vocab.pt'
MODEL_3_MAX_SEQ_LEN = 26
MODEL_3_CONFIDENCE_MODIFIER = 1.0

MODEL_1_CATEGORY_DESCRIPTIONS_MAPPING = './Support_Files/categories_with_category_descriptions.csv'
MODEL_1_LABEL_MAPPING = './Support_Files/label_to_category_mapping.csv'
MODEL_2_CATEGORY_DESCRIPTIONS_MAPPING = './Support_Files/categories_with_category_descriptions.csv'
MODEL_2_LABEL_MAPPING = './Support_Files/label_to_category_mapping.csv'
MODEL_3_CATEGORY_DESCRIPTIONS_MAPPING = './Support_Files/categories_with_category_descriptions.csv'
MODEL_3_LABEL_MAPPING = './Support_Files/label_to_category_mapping.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_input_fpath = ""
_user_text = ""
_predicted_category_1 = ""
_predicted_category_description_1 = ""
_predicted_category_2 = ""
_predicted_category_description_2 = ""
_predicted_category_3 = ""
_predicted_category_description_3 = ""
_manufacturer_flag = False
_brand_flag = False
_packsize_flag = False
_was_storage_provided = False
_is_invalid_file = False
_is_missing_headers = False
_is_usecase_1 = False
_is_usecase_2 = False
_is_usecase_3 = False
_is_more_than_2000 = False
_output_saved_successfully = False
_input_type = ""  # either "manual" or "file"
_checkpoint_cleaned_data = False
_checkpoint_preprocessed_data = False
_checkpoint_model_1 = False
_checkpoint_model_2 = False
_checkpoint_model_3 = False
_checkpoint_category_1 = False
_checkpoint_category_2 = False
_checkpoint_category_3 = False
_checkpoint_output_data = False

_max_seq_length = 26
_dropout = 0.1
_fc_dropout = 0.3
_positional_log_base = 10000.0  # original default value was 10000.0

_dim_term_pos_enc_mean = 512  # dimensions of the model (number of values used to represent single term)
_d_ff_pos_enc_mean = _dim_term_pos_enc_mean * 4  # dimensions used in the feed forward network - the 4 is for q, k, v, o in MultiheadAttention

_num_layers_1 = 6  # number of primary encoder layers the model uses to process the input
_num_heads_1 = 16  # number of perspectives from which the model processes the input

_num_layers_2 = 6  # number of primary encoder layers the model uses to process the input
_num_heads_2 = 16  # number of perspectives from which the model processes the input

_batch_size = 128
_num_batches = 0
_num_labels = 2029  # number of categories in data
_criterion = torch.nn.CrossEntropyLoss()

_prediction_iter = "placeholder"
_text_pipeline = "placeholder"
_prediction_dataloader = "placeholder"
_current_vocabulary = "placeholder"
_current_tokenizer = "placeholder"

"""
Check for inline arguments
"""
_did_receive_arguments = False
_arguments = []
# Check if any inline arguments were provided
if len(sys.argv) > 1:

    # Update the arguments_received flag which determines whether to open the GUI or run in headless mode
    _did_receive_arguments = True
    # Populate the arguments list with the provided arguments
    _arguments = sys.argv[1:]
    # Populate the filename based on the arguments provided
    _input_fpath = _arguments[0]
    print("Input filepath: " + str(_input_fpath))

# If no arguments received, build the GUI
else:

    """
    Configuring GUI
    """

    window = tk.Tk()
    window.title("Product Classifier")

    window.rowconfigure(0, minsize=50, weight=0)
    window.rowconfigure(1, minsize=50, weight=0)
    window.rowconfigure(2, minsize=150, weight=0)
    window.columnconfigure(0, minsize=800, weight=1)

    frm_greeting = tk.Frame(window, bd=0)
    frm_load_button = tk.Frame(window, bd=0)
    frm_progress = tk.Frame(window, bd=0)
    frm_finished = tk.Frame(window, bd=0)
    frm_results = tk.Frame(window, bd=0)
    frm_user_input_banner = tk.Frame(window, bd=0)
    frm_user_input = tk.Frame(window, bd=0)
    frm_user_input_button = tk.Frame(window, bd=0)
    frm_user_input_processing = tk.Frame(window, bd=0)

    frm_greeting.rowconfigure(0, minsize=50, weight=0)
    frm_greeting.columnconfigure(0, minsize=400, weight=1)

    frm_load_button.rowconfigure(0, minsize=50, weight=0)
    frm_load_button.columnconfigure(0, minsize=150, weight=1)
    frm_load_button.columnconfigure(1, minsize=100, weight=0)
    frm_load_button.columnconfigure(2, minsize=150, weight=1)

    frm_progress.rowconfigure(0, minsize=50, weight=0)
    frm_progress.rowconfigure(1, minsize=20, weight=0)
    frm_progress.columnconfigure(0, minsize=400, weight=1)

    frm_finished.rowconfigure(0, minsize=50, weight=0)
    frm_finished.columnconfigure(0, minsize=400, weight=1)

    frm_results.rowconfigure(0, minsize=50, weight=0)
    frm_results.rowconfigure(1, minsize=50, weight=0)
    frm_results.rowconfigure(2, minsize=50, weight=0)
    frm_results.columnconfigure(0, minsize=400, weight=1)

    frm_user_input_banner.rowconfigure(0, minsize=50, weight=0)
    frm_user_input_banner.columnconfigure(0, minsize=400, weight=1)

    frm_user_input.rowconfigure(0, minsize=50, weight=0)
    frm_user_input.rowconfigure(1, minsize=50, weight=0)
    frm_user_input.rowconfigure(2, minsize=50, weight=0)
    frm_user_input.rowconfigure(3, minsize=50, weight=0)
    frm_user_input.rowconfigure(4, minsize=50, weight=0)
    frm_user_input.columnconfigure(0, weight=0)
    frm_user_input.columnconfigure(1, weight=10)

    frm_user_input_processing.rowconfigure(0, minsize=50, weight=0)
    frm_user_input_processing.columnconfigure(0, minsize=400, weight=1)

    frm_user_input_button.rowconfigure(0, minsize=50, weight=0)
    frm_user_input_button.rowconfigure(1, minsize=15, weight=0)
    frm_user_input_button.columnconfigure(0, minsize=150, weight=1)
    frm_user_input_button.columnconfigure(1, minsize=100, weight=0)
    frm_user_input_button.columnconfigure(2, minsize=100, weight=0)
    frm_user_input_button.columnconfigure(3, minsize=150, weight=1)

    lbl_greeting = tk.Label(frm_greeting,
                            text="To load product information from a file, click the button below.",
                            font=("Arial", 12))
    lbl_greeting.grid(row=0, column=0, sticky="nsew", padx=25, pady=10)

    btn_open = tk.Button(frm_load_button, text="Load File", border=4)
    btn_open.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)

    progressbar_ind = Progressbar(frm_progress, mode="indeterminate")
    progressbar_ind.grid(row=0)
    progressbar_det = Progressbar(frm_progress, orient="horizontal", mode="determinate")
    progressbar_ind.grid(row=1)

    lbl_finished = tk.Label(frm_finished, text="Processing completed!  Output file saved in Output_Files folder.",
                            foreground="green", font=("Arial", 12))

    lbl_results_1 = tk.Label(frm_results, text="Category: ", foreground="green", font=("Arial", 12))
    lbl_results_1.grid(row=0, column=0, sticky="nsw", padx=50, pady=10)
    lbl_results_2 = tk.Label(frm_results, text="Category: ", foreground="green", font=("Arial", 12))
    lbl_results_2.grid(row=1, column=0, sticky="nsw", padx=50, pady=10)
    lbl_results_3 = tk.Label(frm_results, text="Category: ", foreground="green", font=("Arial", 12))
    lbl_results_3.grid(row=2, column=0, sticky="nsw", padx=50, pady=10)

    lbl_user_input_banner = tk.Label(frm_user_input_banner,
                                     text="To enter product information manually, use the fields below.",
                                     font=("Arial", 12))
    lbl_user_input_banner.grid(row=0, column=0, sticky="nsew", padx=25, pady=5)

    lbl_description = tk.Label(frm_user_input, text="Description")
    lbl_description.grid(row=0, column=0, sticky="nse", padx=20, pady=5)
    ent_description = tk.Entry(frm_user_input)
    ent_description.grid(row=0, column=1, sticky="nsew", padx=20, pady=12)
    lbl_manufacturer = tk.Label(frm_user_input, text="Manufacturer")
    lbl_manufacturer.grid(row=1, column=0, sticky="nse", padx=20, pady=5)
    ent_manufacturer = tk.Entry(frm_user_input)
    ent_manufacturer.grid(row=1, column=1, sticky="nsew", padx=20, pady=12)
    lbl_brand = tk.Label(frm_user_input, text="Brand")
    lbl_brand.grid(row=2, column=0, sticky="nse", padx=20, pady=5)
    ent_brand = tk.Entry(frm_user_input)
    ent_brand.grid(row=2, column=1, sticky="nsew", padx=20, pady=12)
    lbl_packsize = tk.Label(frm_user_input, text="Pack Size")
    lbl_packsize.grid(row=3, column=0, sticky="nse", padx=20, pady=5)
    ent_packsize = tk.Entry(frm_user_input)
    ent_packsize.grid(row=3, column=1, sticky="nsew", padx=20, pady=12)
    lbl_storage = tk.Label(frm_user_input, text="Storage  (Ex: Frozen)")
    lbl_storage.grid(row=4, column=0, sticky="nse", padx=20, pady=5)
    ent_storage = tk.Entry(frm_user_input)
    ent_storage.grid(row=4, column=1, sticky="nsew", padx=20, pady=12)

    lbl_user_input_processing = tk.Label(frm_user_input_processing, text="Processing...", foreground="red")
    lbl_user_input_processing.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    btn_user_input = tk.Button(frm_user_input_button, text="Submit Data", border=4)
    btn_user_input.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)

    btn_clear_data = tk.Button(frm_user_input_button, text="Clear", border=4)
    btn_clear_data.grid(row=0, column=2, sticky="nsew", padx=10, pady=5)


def log_error(exception):

    global frm_progress
    global frm_finished
    global frm_results
    global lbl_finished
    global lbl_greeting
    global btn_open
    global btn_user_input
    global btn_clear_data
    global _predicted_category_1
    global _predicted_category_2
    global _predicted_category_3
    global _predicted_category_description_1
    global _predicted_category_description_2
    global _predicted_category_description_3
    global ent_description
    global ent_manufacturer
    global ent_brand
    global ent_packsize
    global ent_storage

    log_directory = "C:/Temp/Product_Classifier_Error_Logs/"
    log_filepath = str(log_directory) + 'product_classifier_error_log_' + \
                   str(datetime.now().strftime('%Y-%m-%d_%H.%M.%S')) + '.txt'

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    with open(log_filepath, 'w') as file:
        file.write("------------------------------\n")
        file.write("Product Classifier Error Log\n")
        file.write("------------------------------\n")
        file.write("\n")
        file.write("Datetime: " + str(datetime.now().strftime('%Y-%m-%d_%H.%M.%S')) + "\n")
        file.write("Input Type: " + str(_input_type) + "\n")
        file.write("\n")
        if _input_type == "file":
            file.write("Input Filepath: " + str(_input_fpath) + "\n")
            file.write("\n")
        else:
            description = ent_description.get()
            manufacturer = ent_manufacturer.get()
            brand = ent_brand.get()
            packsize = ent_packsize.get()
            storage = ent_storage.get()
            file.write("Description: " + str(description) + "\n")
            file.write("Manufacturer: " + str(manufacturer) + "\n")
            file.write("Brand: " + str(brand) + "\n")
            file.write("Pack Size: " + str(packsize) + "\n")
            file.write("Storage: " + str(storage) + "\n")
            file.write("\n")
        file.write("Passed Cleaned Data Checkpoint: " + str(_checkpoint_cleaned_data) + "\n")
        file.write("Passed Preprocessed Data Checkpoint: " + str(_checkpoint_preprocessed_data) + "\n")
        file.write("Passed Model 1 Checkpoint: " + str(_checkpoint_model_1) + "\n")
        file.write("Passed Model 2 Checkpoint: " + str(_checkpoint_model_2) + "\n")
        file.write("Passed Model 3 Checkpoint: " + str(_checkpoint_model_3) + "\n")
        file.write("Passed Category 1 Checkpoint: " + str(_checkpoint_category_1) + "\n")
        file.write("Passed Category 2 Checkpoint: " + str(_checkpoint_category_2) + "\n")
        file.write("Passed Category 3 Checkpoint: " + str(_checkpoint_category_3) + "\n")
        file.write("Passed Output Data Checkpoint: " + str(_checkpoint_output_data) + "\n")

        file.write("\n")
        file.write("Exception occurred in process_from_input function." + "\n")
        file.write("Exception: " + str(exception) + "\n")
        file.write("\n")
        file.write("----------\n")
        file.write("Traceback:\n")
        file.write("----------\n")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        formatted_traceback = traceback.format_exception(exc_type, exc_value, exc_traceback)
        for line in formatted_traceback:
            file.write(str(line) + "\n")

    _predicted_category_1 = -6
    _predicted_category_2 = ""
    _predicted_category_3 = ""
    _predicted_category_description_1 = ""
    _predicted_category_description_2 = ""
    _predicted_category_description_3 = ""

    if not _did_receive_arguments:
        progressbar_ind.stop()
        frm_progress.grid_remove()
        frm_results.grid_remove()
        frm_finished.grid(row=0, column=0, sticky="nsew")
        lbl_finished["text"] = "** Unexpected Error **  Please reach out for assistance."
        lbl_greeting["text"] = "To load product information from a file, click the button below."
        lbl_finished["foreground"] = "red"
        btn_open["state"] = "normal"
        btn_user_input["state"] = "normal"
        btn_clear_data["state"] = "normal"


def build_iter(data_list):
    """
    Builds a list containing the description from each record in the given data list.

    :param data_list: (list): A list version of the dataset split
    :return iterable_list (list): list of description text from the given data list
    """

    try:

        iterable_list = []

        # Check to see if input is a single string
        if isinstance(data_list, str):
            iterable_list.append(data_list)

        # Otherwise loop through the items in the list, extract the description, and add it to the iterable_list
        else:
            for i in range(len(data_list.index)):
                text = data_list['description'].iloc[i]
                iterable_list.append(text)

        return iterable_list

    except Exception as e:
        log_error(exception=e)


def init_iterators(pred_list):
    """
    Creates iterable list for the input data.

    :param pred_list: (list): Data list for the input data
    :return None
    """

    global _prediction_iter

    # Create iterable list
    _prediction_iter = build_iter(pred_list)

    return


def tokenize_text(x):

    global _current_vocabulary
    global _current_tokenizer

    try:

        # Experimentation revealed that performing this second lemmatization attempt on unrecognized terms improved results
        split_text = x.split()
        rebuilt_text = ""
        for i in range(len(split_text)):
            vocab_index = _current_vocabulary.__getitem__(split_text[i])
            if vocab_index != 0 and rebuilt_text == "":
                rebuilt_text = split_text[i]
            elif vocab_index != 0 and rebuilt_text != "":
                rebuilt_text = rebuilt_text + " " + split_text[i]
            elif vocab_index == 0 and rebuilt_text == "":
                try:
                    lemma_word = wnl.lemmatize(split_text[i])
                    rebuilt_text = lemma_word
                except:
                    rebuilt_text = split_text[i]
            elif vocab_index == 0 and rebuilt_text != "":
                try:
                    lemma_word = wnl.lemmatize(split_text[i])
                    rebuilt_text = rebuilt_text + " " + lemma_word
                except:
                    rebuilt_text = rebuilt_text + " " + split_text[i]

        tokenized_text = _current_vocabulary(_current_tokenizer(rebuilt_text))

        return tokenized_text

    except Exception as e:
        log_error(exception=e)


def init_pipelines(vcb, tknzr):
    """
    Creates text pipeline to be used with the DataLoaders.

    :param vcb: (TorchText Vocab): The vocabulary to be used
    :param tknzr: (TorchText Tokenizer): The tokenizer to be used
    :return None
    """
    # Create pipeline for text processing
    global _text_pipeline

    try:

        _text_pipeline = lambda x: tokenize_text(x)

        return

    except Exception as e:
        log_error(exception=e)


def collate_batch(batch):
    """
    Collate function to be used with Torch DataLoaders.

    The text_list which this function builds is a list of tensors, each tensor containing an integer
    representation of a word (token). The number of tensors, len(text_list), matches the batch size.

    The function also uses pad_sequence from torch rnn utils to pad each record to the length
    (number of tokens) of the largest record in the batch.  This allows us to capture token relationships
    in the embedding layer for a bi-directional lstm model.

    :param batch: The batch to be collated
    :return text_list, offsets (Torch Tensors): The batched labels, text, and offsets
    """

    global _text_pipeline
    global _batch_size

    try:

        text_list = []

        # Loop through the products in the batch
        for _text in batch:

            # Run the text for a single product through the text_pipeline (tokenize it) and convert the text to a tensor of those tokens
            processed_text = torch.tensor(_text_pipeline(_text), dtype=torch.int64)

            # Append the tokenized text tensor for the single product to the text_list
            text_list.append(processed_text)

        # Pad the first element in text_list up to max seq length to ensure all records padded to max seq length
        text_list[0] = nn.ConstantPad1d((0, _max_seq_length - text_list[0].shape[0]), 0)(text_list[0])

        # Pad each product to the max number of tokens in the batch and convert the text_list to a tensor
        text = pad_sequence(text_list, batch_first=True)

        return text.to(device)

    except Exception as e:
        log_error(exception=e)


def init_dataloaders():
    """
    Creates Torch DataLoaders for each dataset split (train, valid, and test).

    :return None
    """
    global _prediction_dataloader
    global _prediction_iter

    try:

        # Convert the dataset splits to map style datasets
        prediction_dataset = to_map_style_dataset(_prediction_iter)

        # Create DataLoaders to manage batches to send to the model
        _prediction_dataloader = DataLoader(
            prediction_dataset, batch_size=_batch_size, shuffle=False, collate_fn=collate_batch
        )

        return

    except Exception as e:
        log_error(exception=e)


def predict_one(model, loader):
    """
    Predicts the Category for a single product record.

    :param model: The model used for prediction
    :param loader: The DataLoader for prediction
    :return: predicted label (int), predicted probability (float)
    """

    try:

        model.eval()  # set model to evaluation mode

        with torch.no_grad():
            for idx, (text) in enumerate(loader):
                prediction_output = model(text)
                fc_output = prediction_output[0]
                sm_output = prediction_output[1]
                topk_fc_output = torch.topk(fc_output, 3)
                topk_labels = topk_fc_output[1]  # the topk labels
                topk_sm_output = torch.topk(sm_output, 3)
                topk_probabilities = topk_sm_output[0]


        """
        prediction:         index 0 = f1 for labels, index 1 = sm1 for probabilities
            f1:             products
                f1[0]:      possible labels
                f1[1]:      prediction value
        """

        return topk_labels, topk_probabilities

    except Exception as e:
        log_error(exception=e)


def predict_many(model, loader):
    """
    Predicts the Category for multiple product records.

    :param model: The model used for prediction
    :param loader: The DataLoader for prediction
    :return: predicted labels (list), predicted probabilities (list)
    """

    try:

        model.eval()  # set model to evaluation mode

        predicted_labels = []
        predicted_probabilities = []

        with torch.no_grad():
            for idx, (text) in enumerate(loader):
                prediction_output = model(text)
                fc_output = prediction_output[0]
                sm_output = prediction_output[1]
                topk_fc_output = torch.topk(fc_output, 3)
                topk_labels = topk_fc_output[1]  # the topk labels
                topk_sm_output = torch.topk(sm_output, 3)
                topk_probabilities = topk_sm_output[0]
                for label in topk_labels:
                    predicted_labels.append(label)
                for prob in topk_probabilities:
                    predicted_probabilities.append(prob)

        """
        prediction:         index 0 = f1 for labels, index 1 = sm1 for probabilities
            f1:             products
                f1[0]:      possible labels
                f1[1]:      prediction value
        """

        return predicted_labels, predicted_probabilities

    except Exception as e:
        log_error(exception=e)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        global _positional_log_base

        # initialize a tensor to hold the base positional encoding values
        pe = torch.zeros(max_seq_length, d_model)
        # initialize a tensor with values corresponding to each position within max_seq_length
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # initialize a tensor with values based on the _positional_log_base and the value of d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(_positional_log_base) / d_model))

        # calculate d_model number (tensor) of values for each position based on the div_term values
        pe[:, 0::2] = torch.sin(position * div_term)

        # adjust the tensor of values for each position
        pe[:, 1::2] = torch.cos(position * div_term)

        # unsqueeze pe (wraps it in a size 1 outer tensor) and makes it accessible to the forward method
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):

        # slice pe to fit the max number of terms for the current batch, then add the pe values to the input
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class BasicEncoder(nn.Module):
    def __init__(self, dim_term, num_heads, num_layers, d_ff, dropout, num_labels):
        super(BasicEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(dim_term, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        return src_mask

    def forward(self, src, mask):

        src_mask = mask
        enc_output = src

        # send the src_embedded output through the encoder layers one at a time
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        return enc_output


class MultiEncoderPosEncMean(nn.Module):
    def __init__(self, src_vocab_size, max_seq_length, dropout, num_labels):
        super(MultiEncoderPosEncMean, self).__init__()

        global _dim_term_pos_enc_mean
        global _d_ff_pos_enc_mean
        global _num_layers_1
        global _num_heads_1
        global _num_layers_2
        global _num_heads_2

        self.encoder_embedding = nn.Embedding(src_vocab_size, _dim_term_pos_enc_mean)
        self.positional_encoding = PositionalEncoding(_dim_term_pos_enc_mean, max_seq_length)
        self.dropout1 = nn.Dropout(dropout)
        self.encoder_1 = BasicEncoder(_dim_term_pos_enc_mean, _num_heads_1, _num_layers_1, _d_ff_pos_enc_mean, dropout, num_labels)
        self.encoder_2 = BasicEncoder(_dim_term_pos_enc_mean, _num_heads_2, _num_layers_2, _d_ff_pos_enc_mean, dropout, num_labels)
        self.norm1 = nn.LayerNorm(_dim_term_pos_enc_mean)
        self.fc1 = nn.Linear(_dim_term_pos_enc_mean, num_labels)

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        return src_mask

    def forward(self, x):

        x_mask = self.generate_mask(x)

        # send the input data through the positional encoding module where it goes through text embed and pos embed
        # the tensor representation of each term in each product is calculated
        x = self.dropout1(self.positional_encoding(self.encoder_embedding(x)))

        # send the text-embedded positional-encoded data through the three separate encoders
        enc1_x = self.encoder_1(x, x_mask)
        enc2_x = self.encoder_2(x, x_mask)
        x = self.norm1(enc1_x + enc2_x)

        # represent product as a d_model-sized tensor containing the mean values of the tensors from each term
        x = x.mean(dim=1)

        # send the concatenated encoder output through a linear layer for final classification
        output = self.fc1(x)

        sm = torch.nn.functional.softmax(output, dim=-1)

        """
        prediction:         index 0 = f1 for labels, index 1 = sm1 for probabilities
            f1:             products
                f1[0]:      possible labels
                f1[1]:      prediction value
        """

        return output, sm


def convert_label_to_category(labels):
    """
    Converts predicted label(s) to Category ID(s)
    :param labels: (int or list): The predicted label(s)
    :return:
    """

    try:

        categories = []

        category_df = pd.read_csv(MODEL_1_LABEL_MAPPING)
        category_map = {}

        # Populating the category map
        for i in range(category_df.shape[0]):
            label = category_df.loc[i, 'label']
            category = category_df.loc[i, 'category']
            category_map[label] = category

        if isinstance(labels, int):
            try:
                if labels == -2:
                    category = -2
                elif labels == -3:
                    category = -3
                elif labels == -4:
                    category = -4
                elif labels == -5:
                    category = -5
                else:
                    category = category_map[labels]
                return category
            except:
                category = -1
        else:
            # check to see if just a single label (to handle numpy.int64 which wouldn't get caught by previous code)
            try:
                int(labels)
                try:

                    category = category_map[labels]

                    return category
                except:
                    category = -1
                    return category

            # if not a single label
            except:

                # Converting Labels to Category IDs
                for i in range(len(labels)):
                    label = labels[i]
                    try:
                        category = category_map[label]
                    except:
                        category = -1
                    categories.append(category)

            return categories

    except Exception as e:
        log_error(exception=e)


def predict_from_file():
    """
    Predicts the Category for product data loaded from a .CSV file.

    :return:
    """

    global window
    global _input_fpath
    global CONFIDENCE_MIN_1
    global CONFIDENCE_MIN_2
    global CONFIDENCE_MIN_3
    global _current_vocabulary
    global _current_tokenizer
    global _is_invalid_file
    global _is_missing_headers
    global _is_usecase_1
    global _is_usecase_2
    global _is_usecase_3
    global _is_more_than_2000
    global _output_saved_successfully
    global lbl_greeting
    global _did_receive_arguments
    global _checkpoint_cleaned_data
    global _checkpoint_preprocessed_data
    global _checkpoint_model_1
    global _checkpoint_model_2
    global _checkpoint_model_3
    global _checkpoint_category_1
    global _checkpoint_category_2
    global _checkpoint_category_3
    global _checkpoint_output_data
    global MODEL_1_PATH
    global VOCAB_1_PATH
    global MODEL_1_MAX_SEQ_LEN
    global MODEL_2_PATH
    global VOCAB_2_PATH
    global MODEL_2_MAX_SEQ_LEN
    global MODEL_3_PATH
    global VOCAB_3_PATH
    global MODEL_3_MAX_SEQ_LEN

    try:

        CONFIDENCE_MIN_1 = 0.65
        CONFIDENCE_MIN_2 = 0.05
        CONFIDENCE_MIN_3 = 0.05

        confidence_min_two_votes_1 = 0.65
        confidence_min_one_vote_1 = 0.75

        if not _did_receive_arguments:
            window.update_idletasks()

            lbl_greeting["text"] = "Processing data from file..."

            progressbar_det['value'] += 5

        # Prepare the dataset for prediction
        prediction_data, _is_usecase_1, _is_usecase_2, _is_usecase_3 = \
            prepare_dataset(source_fpath=_input_fpath)

        _checkpoint_cleaned_data = True
        _checkpoint_preprocessed_data = True

        if _did_receive_arguments:
            if type(prediction_data) == str and prediction_data == "invalid_input_file":
                sys.exit()

        # Resetting process flow flags
        _is_more_than_2000 = False
        _is_invalid_file = False
        _is_missing_headers = False
        _output_saved_successfully = False

        if not _did_receive_arguments:

            # Check file size
            if len(prediction_data) > 500:
                lbl_greeting["text"] = "File contains more than 500 products. This may take a few minutes..."
            # Currently set to block any files with more than 2000 products
            if len(prediction_data) > 2000:
                print("\nMore than 2000 products.\n")
                _is_more_than_2000 = True
                return

            # Check for invalid files
            if isinstance(prediction_data, str) and prediction_data == "invalid_input_file":
                print("\nInvalid Input File.\n")
                _is_invalid_file = True
                return
            elif isinstance(prediction_data, str) and prediction_data == "missing_headers":
                print("\nInvalid Input File.\n")
                _is_missing_headers = True
                return
            elif not prediction_data.empty:  # reset all flags to False for this condition
                _is_invalid_file = False
                _is_missing_headers = False
                _is_more_than_2000 = False
                _output_saved_successfully = False
            else:
                print("\nInvalid Input File.\n")
                _is_invalid_file = True
                return

        # Initialize iterable lists for each split
        init_iterators(prediction_data)

        # Get a basic english tokenizer to use for custom vocabulary
        _current_tokenizer = get_tokenizer("basic_english")

        if not _did_receive_arguments:
            progressbar_det['value'] += 5

        #####
        # MODEL 1
        #####

        # Load the model's vocabulary
        _current_vocabulary = torch.load(VOCAB_1_PATH)

        # Define the vocab size based on the vocab object
        _model_1_vocab_size = len(_current_vocabulary)

        # Initialize the text pipeline for the dataloader
        init_pipelines(_current_vocabulary, _current_tokenizer)

        # Initialize the DataLoaders for training
        init_dataloaders()

        # Define and load the model
        model_1 = MultiEncoderPosEncMean(_model_1_vocab_size, MODEL_1_MAX_SEQ_LEN, _dropout, num_labels=_num_labels).to(device)
        model_1.load_state_dict(torch.load(MODEL_1_PATH, map_location=torch.device('cpu')))

        if not _did_receive_arguments:
            progressbar_det['value'] += 2

        # Perform the predictions
        predictions_1 = predict_many(model_1, _prediction_dataloader)

        if not _did_receive_arguments:
            progressbar_det['value'] += 2

        labels_1 = []
        probabilities_1 = []

        label_tensors_1 = predictions_1[0]
        for element in label_tensors_1:
            labels = []
            for value in element:
                label = value.item()
                labels.append(label)
            labels_1.append(labels)

        probability_tensors_1 = predictions_1[1]
        for element in probability_tensors_1:
            probabilities = []
            for value in element:
                probability = value.item()
                probability = probability * MODEL_1_CONFIDENCE_MODIFIER
                probabilities.append(probability)
            probabilities_1.append(probabilities)

        if not _did_receive_arguments:
            progressbar_det['value'] += 2

        _checkpoint_model_1 = True

        #####
        # MODEL 2
        #####

        # Load the model's vocabulary
        _current_vocabulary = torch.load(VOCAB_2_PATH)

        # Define the vocab size based on the vocab object
        _model_2_vocab_size = len(_current_vocabulary)

        # Initialize the text pipeline for the dataloader
        init_pipelines(_current_vocabulary, _current_tokenizer)

        # Initialize the DataLoaders for training
        init_dataloaders()

        # Define and load the model
        model_2 = MultiEncoderPosEncMean(_model_2_vocab_size, MODEL_2_MAX_SEQ_LEN, _dropout, num_labels=_num_labels).to(device)
        model_2.load_state_dict(torch.load(MODEL_2_PATH, map_location=torch.device('cpu')))

        if not _did_receive_arguments:
            progressbar_det['value'] += 2

        # Perform the predictions
        predictions_2 = predict_many(model_2, _prediction_dataloader)

        if not _did_receive_arguments:
            progressbar_det['value'] += 2

        labels_2 = []
        probabilities_2 = []

        label_tensors_2 = predictions_2[0]
        for element in label_tensors_2:
            labels = []
            for value in element:
                label = value.item()
                labels.append(label)
            labels_2.append(labels)

        probability_tensors_2 = predictions_2[1]
        for element in probability_tensors_2:
            probabilities = []
            for value in element:
                probability = value.item()
                probability = probability * MODEL_2_CONFIDENCE_MODIFIER
                probabilities.append(probability)
            probabilities_2.append(probabilities)

        if not _did_receive_arguments:
            progressbar_det['value'] += 2

        _checkpoint_model_2 = True

        #####
        # MODEL 3
        #####

        # Load the model's vocabulary
        _current_vocabulary = torch.load(VOCAB_3_PATH)

        # Define the vocab size based on the vocab object
        _model_3_vocab_size = len(_current_vocabulary)

        # Initialize the text pipeline for the dataloader
        init_pipelines(_current_vocabulary, _current_tokenizer)

        # Initialize the DataLoaders for training
        init_dataloaders()

        # Define and load the model
        model_3 = MultiEncoderPosEncMean(_model_3_vocab_size, MODEL_3_MAX_SEQ_LEN, _dropout, num_labels=_num_labels).to(device)
        model_3.load_state_dict(torch.load(MODEL_3_PATH, map_location=torch.device('cpu')))

        if not _did_receive_arguments:
            progressbar_det['value'] += 2

        # Perform the predictions
        predictions_3 = predict_many(model_3, _prediction_dataloader)

        if not _did_receive_arguments:
            progressbar_det['value'] += 2

        labels_3 = []
        probabilities_3 = []

        label_tensors_3 = predictions_3[0]
        for element in label_tensors_3:
            labels = []
            for value in element:
                label = value.item()
                labels.append(label)
            labels_3.append(labels)

        probability_tensors_3 = predictions_3[1]
        for element in probability_tensors_3:
            probabilities = []
            for value in element:
                probability = value.item()
                probability = probability * MODEL_3_CONFIDENCE_MODIFIER
                probabilities.append(probability)
            probabilities_3.append(probabilities)

        if not _did_receive_arguments:
            progressbar_det['value'] += 2

        _checkpoint_model_3 = True

        #####
        # Ensemble Predictions
        #####

        predicted_labels_ensemble_1 = []
        predicted_labels_ensemble_2 = []
        predicted_labels_ensemble_3 = []

        # capture product count to manage progress bar updates in the loop below
        product_count = len(labels_1)

        if not len(labels_1) == len(labels_2) == len(labels_3):
            # print("error")
            return

        else:

            # loop through the products in the current batch
            for i in range(len(labels_1)):

                label_ensemble_1 = -2
                label_ensemble_2 = -2
                label_ensemble_3 = -2

                # use i to manage progress bar updates during this loop
                if product_count >= 10:  # only use iterator to manage progress bar updates if more than 10 products
                    count_to_update_progress = product_count // 10
                    if i % count_to_update_progress == 0:
                        if not _did_receive_arguments:
                            progressbar_det['value'] += 5

                #####
                # Assigning the first Category (best match)
                #####

                # if labels_1[i][0] == labels_2[i][0] and labels_1[i][0] == labels_3[i][0]:
                #     label_ensemble_1 = labels_1[i][0]
                # else:
                #     label_ensemble_1 = -5

                if labels_1[i][0] == labels_2[i][0] and probabilities_1[i][0] > confidence_min_two_votes_1:
                    label_ensemble_1 = labels_1[i][0]
                elif labels_1[i][0] == labels_3[i][0] and probabilities_1[i][0] > confidence_min_two_votes_1:
                    label_ensemble_1 = labels_1[i][0]
                elif labels_2[i][0] == labels_3[i][0] and probabilities_2[i][0] > confidence_min_two_votes_1:
                    label_ensemble_1 = labels_2[i][0]
                elif probabilities_1[i][0] > confidence_min_one_vote_1 or probabilities_2[i][0] > confidence_min_one_vote_1 or \
                        probabilities_3[i][0] > confidence_min_one_vote_1:
                    if probabilities_1[i][0] >= probabilities_2[i][0] and probabilities_1[i][0] >= probabilities_3[i][0]:
                        label_ensemble_1 = labels_1[i][0]
                    elif probabilities_2[i][0] >= probabilities_1[i][0] and probabilities_2[i][0] >= probabilities_3[i][0]:
                        label_ensemble_1 = labels_2[i][0]
                    elif probabilities_3[i][0] >= probabilities_1[i][0] and probabilities_3[i][0] >= probabilities_2[i][0]:
                        label_ensemble_1 = labels_3[i][0]
                else:
                    label_ensemble_1 = -5

                _checkpoint_category_1 = True

                # check to see if we have a confident 1st choice
                if label_ensemble_1 == -5:
                    label_ensemble_2 = -5
                    label_ensemble_3 = -5

                # only execute the following logic if we have a confident 1st and 2nd choice
                else:
                    #####
                    # Assigning the second Category (2nd best match)
                    #####

                    if (label_ensemble_1 != labels_1[i][0] and
                        label_ensemble_1 != labels_2[i][0]) and \
                            probabilities_1[i][0] >= probabilities_2[i][0]:
                        label_ensemble_2 = labels_1[i][0]
                    elif (label_ensemble_1 != labels_1[i][0] and
                          label_ensemble_1 != labels_2[i][0]) and \
                            probabilities_1[i][0] < probabilities_2[i][0]:
                        label_ensemble_2 = labels_2[i][0]
                    elif (label_ensemble_1 != labels_2[i][0] and
                          label_ensemble_1 != labels_3[i][0]) and \
                            probabilities_2[i][0] >= probabilities_3[i][0]:
                        label_ensemble_2 = labels_2[i][0]
                    elif (label_ensemble_1 != labels_2[i][0] and
                          label_ensemble_1 != labels_3[i][0]) and \
                            probabilities_2[i][0] < probabilities_3[i][0]:
                        label_ensemble_2 = labels_3[i][0]
                    elif (label_ensemble_1 != labels_1[i][0] and
                          label_ensemble_1 != labels_3[i][0]) and \
                            probabilities_1[i][0] >= probabilities_3[i][0]:
                        label_ensemble_2 = labels_1[i][0]
                    elif (label_ensemble_1 != labels_1[i][0] and
                          label_ensemble_1 != labels_3[i][0]) and \
                            probabilities_1[i][0] < probabilities_3[i][0]:
                        label_ensemble_2 = labels_3[i][0]
                    elif label_ensemble_1 != labels_1[i][0] and probabilities_1[i][0] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_1[i][0]
                    elif label_ensemble_1 != labels_2[i][0] and probabilities_2[i][0] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_2[i][0]
                    elif label_ensemble_1 != labels_3[i][0] and probabilities_3[i][0] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_3[i][0]
                    elif labels_1[i][0] == labels_2[i][1] and label_ensemble_1 != labels_1[i][0] and probabilities_1[i][
                        0] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_1[i][0]
                    elif labels_1[i][0] == labels_3[i][1] and label_ensemble_1 != labels_1[i][0] and probabilities_1[i][
                        0] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_1[i][0]
                    elif labels_2[i][0] == labels_1[i][1] and label_ensemble_1 != labels_2[i][0] and probabilities_2[i][
                        0] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_2[i][0]
                    elif labels_2[i][0] == labels_3[i][1] and label_ensemble_1 != labels_2[i][0] and probabilities_2[i][
                        0] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_2[i][0]
                    elif labels_3[i][0] == labels_1[i][1] and label_ensemble_1 != labels_3[i][0] and probabilities_3[i][
                        0] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_3[i][0]
                    elif labels_3[i][0] == labels_2[i][1] and label_ensemble_1 != labels_3[i][0] and probabilities_3[i][
                        0] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_3[i][0]
                    elif labels_1[i][1] == labels_2[i][1] and label_ensemble_1 != labels_1[i][1] and probabilities_1[i][
                        1] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_1[i][1]
                    elif labels_1[i][1] == labels_3[i][1] and label_ensemble_1 != labels_1[i][1] and probabilities_1[i][
                        1] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_1[i][1]
                    elif labels_2[i][1] == labels_3[i][1] and label_ensemble_1 != labels_2[i][1] and probabilities_2[i][
                        1] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_2[i][1]
                    elif (probabilities_1[i][1] > CONFIDENCE_MIN_2 or probabilities_2[i][1] > CONFIDENCE_MIN_2 or probabilities_3[i][
                        1] > CONFIDENCE_MIN_2) and \
                            probabilities_1[i][1] >= probabilities_2[i][1] and probabilities_1[i][1] >= probabilities_3[i][1] and \
                            label_ensemble_1 != labels_1[i][1]:
                        label_ensemble_2 = labels_1[i][1]
                    elif (probabilities_1[i][1] > CONFIDENCE_MIN_2 or probabilities_2[i][1] > CONFIDENCE_MIN_2 or probabilities_3[i][
                        1] > CONFIDENCE_MIN_2) and \
                            probabilities_2[i][1] >= probabilities_1[i][1] and probabilities_2[i][1] >= probabilities_3[i][1] and \
                            label_ensemble_1 != labels_2[i][1]:
                        label_ensemble_2 = labels_2[i][1]
                    elif (probabilities_1[i][1] > CONFIDENCE_MIN_2 or probabilities_2[i][1] > CONFIDENCE_MIN_2 or probabilities_3[i][
                        1] > CONFIDENCE_MIN_2) and \
                            probabilities_3[i][1] >= probabilities_1[i][1] and probabilities_3[i][1] >= probabilities_2[i][1] and \
                            label_ensemble_1 != labels_3[i][1]:
                        label_ensemble_2 = labels_3[i][1]
                    elif labels_1[i][1] == labels_2[i][2] and label_ensemble_1 != labels_1[i][1] and probabilities_1[i][
                        1] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_1[i][1]
                    elif labels_1[i][1] == labels_3[i][2] and label_ensemble_1 != labels_1[i][1] and probabilities_1[i][
                        1] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_1[i][1]
                    elif labels_2[i][1] == labels_1[i][2] and label_ensemble_1 != labels_2[i][1] and probabilities_2[i][
                        1] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_2[i][1]
                    elif labels_2[i][1] == labels_3[i][2] and label_ensemble_1 != labels_2[i][1] and probabilities_2[i][
                        1] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_2[i][1]
                    elif labels_3[i][1] == labels_1[i][2] and label_ensemble_1 != labels_3[i][1] and probabilities_3[i][
                        1] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_3[i][1]
                    elif labels_3[i][1] == labels_2[i][2] and label_ensemble_1 != labels_3[i][1] and probabilities_3[i][
                        1] > CONFIDENCE_MIN_2:
                        label_ensemble_2 = labels_3[i][1]
                    else:
                        if probabilities_1[i][1] >= probabilities_2[i][1] and probabilities_1[i][1] >= probabilities_3[i][1] and \
                                label_ensemble_1 != labels_1[i][1] and probabilities_1[i][1] > CONFIDENCE_MIN_2:
                            label_ensemble_2 = labels_1[i][1]
                        elif probabilities_2[i][1] >= probabilities_1[i][1] and probabilities_2[i][1] >= probabilities_3[i][1] and \
                                label_ensemble_1 != labels_2[i][1] and probabilities_2[i][1] > CONFIDENCE_MIN_2:
                            label_ensemble_2 = labels_2[i][1]
                        elif probabilities_3[i][1] >= probabilities_1[i][1] and probabilities_3[i][1] >= probabilities_2[i][1] and \
                                label_ensemble_1 != labels_3[i][1] and probabilities_3[i][1] > CONFIDENCE_MIN_2:
                            label_ensemble_2 = labels_3[i][1]
                        else:
                            label_ensemble_2 = -5

                _checkpoint_category_2 = True

                # check to see if we have a confident 1st and 2nd choice
                if label_ensemble_1 == -5 or label_ensemble_2 == -5:
                    label_ensemble_2 = -5
                    label_ensemble_3 = -5
                # only execute the following logic if we have a confident 1st and 2nd choice
                else:
                    #####
                    # Assigning the third Category (3rd best match)
                    #####

                    if labels_1[i][0] == labels_2[i][1] and label_ensemble_1 != labels_1[i][0] and \
                            label_ensemble_2 != labels_1[i][0] and probabilities_1[i][0] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_1[i][0]
                    elif labels_1[i][0] == labels_3[i][1] and label_ensemble_1 != labels_1[i][0] and \
                            label_ensemble_2 != labels_1[i][0] and probabilities_1[i][0] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_1[i][0]
                    elif labels_2[i][0] == labels_1[i][1] and label_ensemble_1 != labels_2[i][0] and \
                            label_ensemble_2 != labels_2[i][0] and probabilities_2[i][0] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_2[i][0]
                    elif labels_2[i][0] == labels_3[i][1] and label_ensemble_1 != labels_2[i][0] and \
                            label_ensemble_2 != labels_2[i][0] and probabilities_2[i][0] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_2[i][0]
                    elif labels_3[i][0] == labels_1[i][1] and label_ensemble_1 != labels_3[i][0] and \
                            label_ensemble_2 != labels_3[i][0] and probabilities_3[i][0] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_3[i][0]
                    elif labels_3[i][0] == labels_2[i][1] and label_ensemble_1 != labels_3[i][0] and \
                            label_ensemble_2 != labels_3[i][0] and probabilities_3[i][0] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_3[i][0]
                    elif labels_1[i][1] == labels_2[i][1] and label_ensemble_1 != labels_1[i][1] and \
                            label_ensemble_2 != labels_1[i][1] and probabilities_1[i][1] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_1[i][1]
                    elif labels_1[i][1] == labels_3[i][1] and label_ensemble_1 != labels_1[i][1] and \
                            label_ensemble_2 != labels_1[i][1] and probabilities_1[i][1] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_1[i][1]
                    elif labels_2[i][1] == labels_3[i][1] and label_ensemble_1 != labels_2[i][1] and \
                            label_ensemble_2 != labels_2[i][1] and probabilities_2[i][1] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_2[i][1]
                    elif label_ensemble_1 != labels_1[i][0] and \
                            label_ensemble_2 != labels_1[i][0] and \
                            probabilities_1[i][0] > probabilities_2[i][0] and probabilities_1[i][0] > probabilities_2[i][1] and \
                            probabilities_1[i][0] > probabilities_3[i][0] and probabilities_1[i][0] > probabilities_3[i][1] and \
                            probabilities_1[i][0] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_1[i][0]
                    elif label_ensemble_1 != labels_2[i][0] and \
                            label_ensemble_2 != labels_2[i][0] and \
                            probabilities_2[i][0] > probabilities_1[i][0] and probabilities_2[i][0] > probabilities_1[i][1] and \
                            probabilities_2[i][0] > probabilities_3[i][0] and probabilities_2[i][0] > probabilities_3[i][1] and \
                            probabilities_2[i][0] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_2[i][0]
                    elif label_ensemble_1 != labels_3[i][0] and \
                            label_ensemble_2 != labels_3[i][0] and \
                            probabilities_3[i][0] > probabilities_1[i][0] and probabilities_3[i][0] > probabilities_1[i][1] and \
                            probabilities_3[i][0] > probabilities_2[i][0] and probabilities_3[i][0] > probabilities_2[i][1] and \
                            probabilities_3[i][0] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_3[i][0]
                    elif label_ensemble_1 != labels_1[i][1] and \
                            label_ensemble_2 != labels_1[i][1] and \
                            probabilities_1[i][1] > probabilities_2[i][0] and probabilities_1[i][1] > probabilities_2[i][1] and \
                            probabilities_1[i][1] > probabilities_3[i][0] and probabilities_1[i][1] > probabilities_3[i][1] and \
                            probabilities_1[i][1] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_1[i][1]
                    elif label_ensemble_1 != labels_2[i][1] and \
                            label_ensemble_2 != labels_2[i][1] and \
                            probabilities_2[i][1] > probabilities_1[i][0] and probabilities_2[i][1] > probabilities_1[i][1] and \
                            probabilities_2[i][1] > probabilities_3[i][0] and probabilities_2[i][1] > probabilities_3[i][1] and \
                            probabilities_2[i][1] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_2[i][1]
                    elif label_ensemble_1 != labels_3[i][1] and \
                            label_ensemble_2 != labels_3[i][1] and \
                            probabilities_3[i][1] > probabilities_1[i][0] and probabilities_3[i][1] > probabilities_1[i][1] and \
                            probabilities_3[i][1] > probabilities_2[i][0] and probabilities_3[i][1] > probabilities_2[i][1] and \
                            probabilities_3[i][1] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_3[i][1]
                    elif label_ensemble_1 != labels_1[i][2] and \
                            label_ensemble_2 != labels_1[i][2] and \
                            probabilities_1[i][2] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_1[i][2]
                    elif label_ensemble_1 != labels_2[i][2] and \
                            label_ensemble_2 != labels_2[i][2] and \
                            probabilities_2[i][2] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_2[i][2]
                    elif label_ensemble_1 != labels_3[i][2] and \
                            label_ensemble_2 != labels_3[i][2] and \
                            probabilities_3[i][2] > CONFIDENCE_MIN_3:
                        label_ensemble_3 = labels_3[i][2]
                    else:
                        label_ensemble_3 = -5

                    if labels_1[i][0] == labels_2[i][0] and labels_1[i][0] == labels_3[i][0]:
                        label_ensemble_1 = labels_1[i][0]

                    if label_ensemble_1 == -3:
                        label_ensemble_3 = -5
                    elif label_ensemble_2 == -5 and label_ensemble_3 != -5 and label_ensemble_1 != label_ensemble_3:
                        label_ensemble_2 = label_ensemble_3
                        label_ensemble_3 = -5

                _checkpoint_category_3 = True

                # Append the Category IDs to the predicted categories lists
                predicted_labels_ensemble_1.append(label_ensemble_1)
                predicted_labels_ensemble_2.append(label_ensemble_2)
                predicted_labels_ensemble_3.append(label_ensemble_3)

            if not _did_receive_arguments:
                progressbar_det['value'] += 5

            # Convert the labels to Category IDs
            predicted_categories_ensemble_1 = convert_label_to_category(predicted_labels_ensemble_1)
            predicted_categories_ensemble_2 = convert_label_to_category(predicted_labels_ensemble_2)
            predicted_categories_ensemble_3 = convert_label_to_category(predicted_labels_ensemble_3)

            # Get the Category Descriptions for all the products based on the Category IDs
            predicted_category_descs_ensemble_1 = get_category_description_load(predicted_categories_ensemble_1)
            predicted_category_descs_ensemble_2 = get_category_description_load(predicted_categories_ensemble_2)
            predicted_category_descs_ensemble_3 = get_category_description_load(predicted_categories_ensemble_3)

            if not _did_receive_arguments:
                progressbar_det['value'] += 5

            # Convert the lists to Pandas Series
            predicted_categories_series_ensemble_1 = pd.Series(predicted_categories_ensemble_1)
            predicted_category_descs_series_ensemble_1 = pd.Series(predicted_category_descs_ensemble_1)
            predicted_categories_series_ensemble_2 = pd.Series(predicted_categories_ensemble_2)
            predicted_category_descs_series_ensemble_2 = pd.Series(predicted_category_descs_ensemble_2)
            predicted_categories_series_ensemble_3 = pd.Series(predicted_categories_ensemble_3)
            predicted_category_descs_series_ensemble_3 = pd.Series(predicted_category_descs_ensemble_3)

            if not _did_receive_arguments:
                progressbar_det['value'] += 5

            # Create a DataFrame to hold the prediction output data
            predicted_categories_df = pd.DataFrame(columns=['category_1', 'category_description_1',
                                                        'category_2', 'category_description_2',
                                                        'category_3', 'category_description_3'])

            # Populate the DataFrame
            predicted_categories_df['category_1'] = predicted_categories_series_ensemble_1
            predicted_categories_df['category_description_1'] = predicted_category_descs_series_ensemble_1
            predicted_categories_df['category_2'] = predicted_categories_series_ensemble_2
            predicted_categories_df['category_description_2'] = predicted_category_descs_series_ensemble_2
            predicted_categories_df['category_3'] = predicted_categories_series_ensemble_3
            predicted_categories_df['category_description_3'] = predicted_category_descs_series_ensemble_3

            if not _did_receive_arguments:
                progressbar_det['value'] += 5

            # Split the full input filepath to help with output filename, file format, and output folders
            split_fpath = _input_fpath.split("/")
            fname_only = split_fpath[len(split_fpath) - 1]
            input_folder = _input_fpath
            text_to_remove = "/" + str(fname_only)
            while text_to_remove in input_folder:
                input_folder = input_folder.replace(text_to_remove, '')
            split_fname = fname_only.split(".")
            file_extension = "." + str(split_fname[len(split_fname) - 1])
            while file_extension in fname_only:
                fname_only = fname_only.replace(file_extension, '')

            if _is_usecase_1:

                # Prepare the output file for use with the Use Case 1 Template
                predicted_categories_df['category_1'] = predicted_categories_df['category_1'].astype(str)
                predicted_categories_df['category_2'] = predicted_categories_df['category_2'].astype(str)
                predicted_categories_df['category_3'] = predicted_categories_df['category_3'].astype(str)
                predicted_categories_df['category_id_desc_1'] = predicted_categories_df['category_1'].str.cat(predicted_categories_df['category_description_1'], sep='-')
                predicted_categories_df['category_id_desc_2'] = predicted_categories_df['category_2'].str.cat(predicted_categories_df['category_description_2'], sep='-')
                predicted_categories_df['category_id_desc_3'] = predicted_categories_df['category_3'].str.cat(predicted_categories_df['category_description_3'], sep='-')
                predicted_categories_df.drop(['category_1', 'category_description_1', 'category_2',
                                          'category_description_2', 'category_3', 'category_description_3'], axis=1,
                                         inplace=True)
                # Save the output file
                predicted_categories_df.to_excel(input_folder + '/CLASSIFIER_OUTPUT_' +
                                             fname_only + '.xlsx', sheet_name='Categories', index=False)
            elif _is_usecase_2:

                prodnum_series = pd.Series(prediction_data['Product Number'])
                predicted_categories_df['product_number'] = prodnum_series

                # Save the output file
                predicted_categories_df.to_excel(input_folder + '/CLASSIFIER_OUTPUT_' +
                                             fname_only + '.xlsx', sheet_name='Categories', index=False)

            elif _is_usecase_3:

                gtin_series = pd.Series(prediction_data['GTIN'])
                mpn_series = pd.Series(prediction_data['Manufacturer Product Number'])

                predicted_categories_df.insert(0, 'gtin', gtin_series)
                predicted_categories_df.insert(1, 'manufacturer_product_number', mpn_series)

                # Save the output file
                predicted_categories_df.to_excel(input_folder + '/CLASSIFIER_OUTPUT_' +
                                             fname_only + '.xlsx', sheet_name='Categories', index=False)

            # otherwise, check to see if the classifier was run with inline arguments
            else:
                # if did not meet any previous conditions but did receive arguments, save output
                if _did_receive_arguments:
                    # Save the prediction output data as a .csv in the Output_Files folder
                    predicted_categories_df.to_csv(path_or_buf='./Output_Files/predicted_categories_' +
                                                           str(datetime.now().strftime('%Y-%m-%d_%H.%M.%S')) +
                                                           '.csv', sep=',', encoding='utf-8', index=False)
                # otherwise advance the progress bar and return without updating the _output_saves_successfully flag to True
                elif not _did_receive_arguments:
                    progressbar_det['value'] += 20

                    if product_count < 10:
                        progressbar_det['value'] += 50

                    return

            if not _did_receive_arguments:
                progressbar_det['value'] += 20

            if product_count < 10:
                progressbar_det['value'] += 50

            _output_saved_successfully = True
            _checkpoint_output_data = True

            return

    except Exception as e:
        log_error(exception=e)


def predict_from_input():
    """
    Predicts the Category for product data entered through GUI from user.

    :return:
    """

    global window
    global _input_fpath
    global _user_text
    global _predicted_category_1
    global _predicted_category_description_1
    global _predicted_category_2
    global _predicted_category_description_2
    global _predicted_category_3
    global _predicted_category_description_3
    global CONFIDENCE_MIN_1
    global CONFIDENCE_MIN_2
    global CONFIDENCE_MIN_3
    global _manufacturer_flag
    global _brand_flag
    global _packsize_flag
    global _was_storage_provided
    global _current_vocabulary
    global _current_tokenizer
    global ent_description
    global frm_results
    global lbl_results_1
    global lbl_results_2
    global lbl_results_3
    global _input_type
    global _checkpoint_model_1
    global _checkpoint_model_2
    global _checkpoint_model_3
    global _checkpoint_category_1
    global _checkpoint_category_2
    global _checkpoint_category_3
    global _checkpoint_output_data
    global MODEL_1_PATH
    global VOCAB_1_PATH
    global MODEL_2_PATH
    global VOCAB_2_PATH
    global MODEL_3_PATH
    global VOCAB_3_PATH

    try:

        window.update_idletasks()

        _input_type = "manual"

        description = ent_description.get()

        if description == "":

            _predicted_category_1 = -1
            _predicted_category_2 = ""
            _predicted_category_3 = ""
            _predicted_category_description_1 = ""
            _predicted_category_description_2 = ""
            _predicted_category_description_3 = ""
            return

        # Initialize iterable lists for each split
        init_iterators(_user_text)

        # Get a basic english tokenizer to use for custom vocabulary
        _current_tokenizer = get_tokenizer("basic_english")

        # Load the model's vocabulary
        _current_vocabulary = torch.load(VOCAB_1_PATH)

        # Check to see if there are enough recognized words for a reasonable prediction
        recognized_term_counter = 0
        split_text = _user_text.split()
        rebuilt_text = ""
        for i in range(len(split_text)):
            vocab_index = _current_vocabulary.__getitem__(split_text[i])
            if vocab_index != 0 and rebuilt_text == "":
                rebuilt_text = split_text[i]
            elif vocab_index != 0 and rebuilt_text != "":
                rebuilt_text = rebuilt_text + " " + split_text[i]
            elif vocab_index == 0 and rebuilt_text == "":
                try:
                    lemma_word = wnl.lemmatize(split_text[i])
                    rebuilt_text = lemma_word
                except:
                    rebuilt_text = split_text[i]
            elif vocab_index == 0 and rebuilt_text != "":
                try:
                    lemma_word = wnl.lemmatize(split_text[i])
                    rebuilt_text = rebuilt_text + " " + lemma_word
                except:
                    rebuilt_text = rebuilt_text + " " + split_text[i]

        for i in range(len(split_text)):
            vocab_index = _current_vocabulary.__getitem__(rebuilt_text[i])
            if vocab_index != 0:
                recognized_term_counter += 1

        # If there are fewer than 2 recognized terms...
        if recognized_term_counter < 2:
            _predicted_category_1 = -2
            _predicted_category_description_1 = "n/a"
            return _predicted_category_1, _predicted_category_description_1

        else:

            if recognized_term_counter == 2:
                CONFIDENCE_MIN_1 = 0.9
                CONFIDENCE_MIN_2 = 0.05
                CONFIDENCE_MIN_3 = 0.05
            elif recognized_term_counter == 3:
                CONFIDENCE_MIN_1 = 0.75
                CONFIDENCE_MIN_2 = 0.05
                CONFIDENCE_MIN_3 = 0.05
            elif recognized_term_counter == 4:
                CONFIDENCE_MIN_1 = 0.45
                CONFIDENCE_MIN_2 = 0.05
                CONFIDENCE_MIN_3 = 0.05
            elif recognized_term_counter == 5:
                CONFIDENCE_MIN_1 = 0.3
                CONFIDENCE_MIN_2 = 0.05
                CONFIDENCE_MIN_3 = 0.05
            elif recognized_term_counter == 6:
                CONFIDENCE_MIN_1 = 0.3
                CONFIDENCE_MIN_2 = 0.05
                CONFIDENCE_MIN_3 = 0.05
            elif recognized_term_counter == 7:
                CONFIDENCE_MIN_1 = 0.3
                CONFIDENCE_MIN_2 = 0.05
                CONFIDENCE_MIN_3 = 0.05
            elif recognized_term_counter == 8:
                CONFIDENCE_MIN_1 = 0.3
                CONFIDENCE_MIN_2 = 0.05
                CONFIDENCE_MIN_3 = 0.05
            else:
                CONFIDENCE_MIN_1 = 0.3
                CONFIDENCE_MIN_2 = 0.1
                CONFIDENCE_MIN_3 = 0.1

            progressbar_det['value'] += 5

            #####
            # MODEL 1
            #####

            # Define the vocab size based on the vocab object
            _model_1_vocab_size = len(_current_vocabulary)

            # Initialize the text pipeline for the dataloader
            init_pipelines(_current_vocabulary, _current_tokenizer)

            # Initialize the DataLoaders for training
            init_dataloaders()

            # Define and load the first model
            model_1 = MultiEncoderPosEncMean(_model_1_vocab_size, _max_seq_length, _dropout, num_labels=_num_labels).to(device)
            model_1.load_state_dict(torch.load(MODEL_1_PATH, map_location=torch.device('cpu')))

            # Perform the predictions
            prediction_1 = predict_one(model_1, _prediction_dataloader)

            labels_1 = []
            probabilities_1 = []
            predicted_categories_1 = []

            label_tensors_1 = prediction_1[0][0]
            for element in label_tensors_1:
                label = element.item()
                labels_1.append(label)
            probability_tensors_1 = list(prediction_1[1][0])
            for element in probability_tensors_1:
                prob = element.item()
                prob = prob * MODEL_1_CONFIDENCE_MODIFIER
                probabilities_1.append(prob)
            for label in labels_1:
                category = convert_label_to_category(label)
                predicted_categories_1.append(category)

            progressbar_det['value'] += 5

            _checkpoint_model_1 = True

            #####
            # MODEL 2
            #####

            # Load the model's vocabulary
            _current_vocabulary = torch.load(VOCAB_2_PATH)

            # Define the vocab size based on the vocab object
            _model_2_vocab_size = len(_current_vocabulary)

            # Initialize the text pipeline for the dataloader
            init_pipelines(_current_vocabulary, _current_tokenizer)

            # Initialize the DataLoaders for training
            init_dataloaders()

            # Define and load the model
            model_2 = MultiEncoderPosEncMean(_model_2_vocab_size, _max_seq_length, _dropout, num_labels=_num_labels).to(device)
            model_2.load_state_dict(torch.load(MODEL_2_PATH, map_location=torch.device('cpu')))

            # Perform the predictions
            prediction_2 = predict_one(model_2, _prediction_dataloader)

            labels_2 = []
            probabilities_2 = []
            predicted_categories_2 = []

            label_tensors_2 = prediction_2[0][0]
            for element in label_tensors_2:
                label = element.item()
                labels_2.append(label)
            probability_tensors_2 = list(prediction_2[1][0])
            for element in probability_tensors_2:
                prob = element.item()
                prob = prob * MODEL_2_CONFIDENCE_MODIFIER
                probabilities_2.append(prob)
            for label in labels_2:
                category = convert_label_to_category(label)
                predicted_categories_2.append(category)

            progressbar_det['value'] += 5

            _checkpoint_model_2 = True

            #####
            # MODEL 3
            #####

            # Initialize iterable list for 3rd model
            init_iterators(_user_text)

            # Load the model's vocabulary
            _current_vocabulary = torch.load(VOCAB_3_PATH)

            # Define the vocab size based on the vocab object
            _model_3_vocab_size = len(_current_vocabulary)

            # Initialize the text pipeline for the dataloader
            init_pipelines(_current_vocabulary, _current_tokenizer)

            # Initialize the DataLoaders for training
            init_dataloaders()

            # Define and load the model
            model_3 = MultiEncoderPosEncMean(_model_3_vocab_size, _max_seq_length, _dropout, num_labels=_num_labels).to(device)
            model_3.load_state_dict(torch.load(MODEL_3_PATH, map_location=torch.device('cpu')))

            # Perform the predictions
            prediction_3 = predict_one(model_3, _prediction_dataloader)

            labels_3 = []
            probabilities_3 = []
            predicted_categories_3 = []

            label_tensors_3 = prediction_3[0][0]
            for element in label_tensors_3:
                label = element.item()
                labels_3.append(label)
            probability_tensors_3 = list(prediction_3[1][0])
            for element in probability_tensors_3:
                prob = element.item()
                prob = prob * MODEL_3_CONFIDENCE_MODIFIER
                probabilities_3.append(prob)
            for label in labels_3:
                category = convert_label_to_category(label)
                predicted_categories_3.append(category)

            _checkpoint_model_3 = True

            #####
            # Ensemble Predictions
            #####

            progressbar_det['value'] += 5

            predicted_category_ensemble_1 = -2

            #####
            # Assigning the first Category (best match)
            ###
            if labels_1[0] == labels_2[0] and probabilities_1[0] > CONFIDENCE_MIN_1:
                predicted_category_ensemble_1 = predicted_categories_1[0]
            elif labels_1[0] == labels_3[0] and probabilities_1[0] > CONFIDENCE_MIN_1:
                predicted_category_ensemble_1 = predicted_categories_1[0]
            elif labels_2[0] == labels_3[0] and probabilities_2[0] > CONFIDENCE_MIN_1:
                predicted_category_ensemble_1 = predicted_categories_2[0]
            elif probabilities_1[0] > CONFIDENCE_MIN_1 or probabilities_2[0] > CONFIDENCE_MIN_1 or \
                    probabilities_3[0] > CONFIDENCE_MIN_1:
                if probabilities_1[0] >= probabilities_2[0] and probabilities_1[0] >= probabilities_3[0]:
                    predicted_category_ensemble_1 = predicted_categories_1[0]
                elif probabilities_2[0] >= probabilities_1[0] and probabilities_2[0] >= probabilities_3[0]:
                    predicted_category_ensemble_1 = predicted_categories_2[0]
                elif probabilities_3[0] >= probabilities_1[0] and probabilities_3[0] >= probabilities_2[0]:
                    predicted_category_ensemble_1 = predicted_categories_3[0]
            elif labels_1[0] == labels_2[1] and probabilities_1[0] > CONFIDENCE_MIN_1:
                predicted_category_ensemble_1 = predicted_categories_1[0]
            elif labels_1[0] == labels_3[1] and probabilities_1[0] > CONFIDENCE_MIN_1:
                predicted_category_ensemble_1 = predicted_categories_1[0]
            elif labels_2[0] == labels_1[1] and probabilities_2[0] > CONFIDENCE_MIN_1:
                predicted_category_ensemble_1 = predicted_categories_2[0]
            elif labels_2[0] == labels_3[1] and probabilities_2[0] > CONFIDENCE_MIN_1:
                predicted_category_ensemble_1 = predicted_categories_2[0]
            elif labels_3[0] == labels_1[1] and probabilities_3[0] > CONFIDENCE_MIN_1:
                predicted_category_ensemble_1 = predicted_categories_3[0]
            elif labels_3[0] == labels_2[1] and probabilities_3[0] > CONFIDENCE_MIN_1:
                predicted_category_ensemble_1 = predicted_categories_3[0]
            else:
                if probabilities_1[0] >= probabilities_2[0] and probabilities_1[0] >= probabilities_3[0] and \
                        probabilities_1[0] > CONFIDENCE_MIN_1:
                    predicted_category_ensemble_1 = predicted_categories_1[0]
                elif probabilities_2[0] >= probabilities_1[0] and probabilities_2[0] >= probabilities_3[0] and \
                        probabilities_2[0] > CONFIDENCE_MIN_1:
                    predicted_category_ensemble_1 = predicted_categories_2[0]
                elif probabilities_3[0] >= probabilities_1[0] and probabilities_3[0] >= probabilities_2[0] and \
                        probabilities_3[0] > CONFIDENCE_MIN_1:
                    predicted_category_ensemble_1 = predicted_categories_3[0]
                else:
                    predicted_category_ensemble_1 = -3

            progressbar_det['value'] += 30

            _checkpoint_category_1 = True

            #####
            # Assigning the second Category (2nd best match)
            ###
            if (predicted_category_ensemble_1 != predicted_categories_1[0] and
                predicted_category_ensemble_1 != predicted_categories_2[0]) and \
                    probabilities_1[0] >= probabilities_2[0] and probabilities_1[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_1[0]
            elif (predicted_category_ensemble_1 != predicted_categories_1[0] and
                predicted_category_ensemble_1 != predicted_categories_2[0]) and \
                    probabilities_1[0] < probabilities_2[0]  and probabilities_2[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_2[0]
            elif (predicted_category_ensemble_1 != predicted_categories_2[0] and
                predicted_category_ensemble_1 != predicted_categories_3[0]) and \
                    probabilities_2[0] >= probabilities_3[0 and probabilities_2[0] > CONFIDENCE_MIN_2]:
                predicted_category_ensemble_2 = predicted_categories_2[0]
            elif (predicted_category_ensemble_1 != predicted_categories_2[0] and
                predicted_category_ensemble_1 != predicted_categories_3[0]) and \
                    probabilities_2[0] < probabilities_3[0] and probabilities_3[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_3[0]
            elif (predicted_category_ensemble_1 != predicted_categories_1[0] and
                predicted_category_ensemble_1 != predicted_categories_3[0]) and \
                    probabilities_1[0] >= probabilities_3[0] and probabilities_1[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_1[0]
            elif (predicted_category_ensemble_1 != predicted_categories_1[0] and
                predicted_category_ensemble_1 != predicted_categories_3[0]) and \
                    probabilities_1[0] < probabilities_3[0] and probabilities_3[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_1 = predicted_categories_3[0]
            elif predicted_category_ensemble_1 != predicted_categories_1[0] and probabilities_1[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_1[0]
            elif predicted_category_ensemble_1 != predicted_categories_2[0] and probabilities_2[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_2[0]
            elif predicted_category_ensemble_1 != predicted_categories_3[0] and probabilities_3[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_3[0]
            elif labels_1[0] == labels_2[1] and predicted_category_ensemble_1 != predicted_categories_1[0] and probabilities_1[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_1[0]
            elif labels_1[0] == labels_3[1] and predicted_category_ensemble_1 != predicted_categories_1[0] and probabilities_1[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_1[0]
            elif labels_2[0] == labels_1[1] and predicted_category_ensemble_1 != predicted_categories_2[0] and probabilities_2[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_2[0]
            elif labels_2[0] == labels_3[1] and predicted_category_ensemble_1 != predicted_categories_2[0] and probabilities_2[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_2[0]
            elif labels_3[0] == labels_1[1] and predicted_category_ensemble_1 != predicted_categories_3[0] and probabilities_3[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_3[0]
            elif labels_3[0] == labels_2[1] and predicted_category_ensemble_1 != predicted_categories_3[0] and probabilities_3[0] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_3[0]
            elif labels_1[1] == labels_2[1] and predicted_category_ensemble_1 != predicted_categories_1[1] and probabilities_1[1] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_1[1]
            elif labels_1[1] == labels_3[1] and predicted_category_ensemble_1 != predicted_categories_1[1] and probabilities_1[1] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_1[1]
            elif labels_2[1] == labels_3[1] and predicted_category_ensemble_1 != predicted_categories_2[1] and probabilities_2[1] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_2[1]
            elif (probabilities_1[1] > CONFIDENCE_MIN_2 or probabilities_2[1] > CONFIDENCE_MIN_2 or probabilities_3[1] > CONFIDENCE_MIN_2) and \
                probabilities_1[1] >= probabilities_2[1] and probabilities_1[1] >= probabilities_3[1] and \
                predicted_category_ensemble_1 != predicted_categories_1[1]:
                predicted_category_ensemble_2 = predicted_categories_1[1]
            elif (probabilities_1[1] > CONFIDENCE_MIN_2 or probabilities_2[1] > CONFIDENCE_MIN_2 or probabilities_3[1] > CONFIDENCE_MIN_2) and \
                probabilities_2[1] >= probabilities_1[1] and probabilities_2[1] >= probabilities_3[1] and \
                predicted_category_ensemble_1 != predicted_categories_2[1]:
                predicted_category_ensemble_2 = predicted_categories_2[1]
            elif (probabilities_1[1] > CONFIDENCE_MIN_2 or probabilities_2[1] > CONFIDENCE_MIN_2 or probabilities_3[1] > CONFIDENCE_MIN_2) and \
                probabilities_3[1] >= probabilities_1[1] and probabilities_3[1] >= probabilities_2[1] and \
                predicted_category_ensemble_1 != predicted_categories_3[1]:
                predicted_category_ensemble_2 = predicted_categories_3[1]
            elif labels_1[1] == labels_2[2] and predicted_category_ensemble_1 != predicted_categories_1[1] and probabilities_1[1] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_1[1]
            elif labels_1[1] == labels_3[2] and predicted_category_ensemble_1 != predicted_categories_1[1] and probabilities_1[1] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_1[1]
            elif labels_2[1] == labels_1[2] and predicted_category_ensemble_1 != predicted_categories_2[1] and probabilities_2[1] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_2[1]
            elif labels_2[1] == labels_3[2] and predicted_category_ensemble_1 != predicted_categories_2[1] and probabilities_2[1] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_2[1]
            elif labels_3[1] == labels_1[2] and predicted_category_ensemble_1 != predicted_categories_3[1] and probabilities_3[1] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_3[1]
            elif labels_3[1] == labels_2[2] and predicted_category_ensemble_1 != predicted_categories_3[1] and probabilities_3[1] > CONFIDENCE_MIN_2:
                predicted_category_ensemble_2 = predicted_categories_3[1]
            else:
                if probabilities_1[1] >= probabilities_2[1] and probabilities_1[1] >= probabilities_3[1] and \
                        predicted_category_ensemble_1 != predicted_categories_1[1] and probabilities_1[1] > CONFIDENCE_MIN_2:
                    predicted_category_ensemble_2 = predicted_categories_1[1]
                elif probabilities_2[1] >= probabilities_1[1] and probabilities_2[1] >= probabilities_3[1] and \
                        predicted_category_ensemble_1 != predicted_categories_2[1] and probabilities_2[1] > CONFIDENCE_MIN_2:
                    predicted_category_ensemble_2 = predicted_categories_2[1]
                elif probabilities_3[1] >= probabilities_1[1] and probabilities_3[1] >= probabilities_2[1] and \
                        predicted_category_ensemble_1 != predicted_categories_3[1] and probabilities_3[1] > CONFIDENCE_MIN_2:
                    predicted_category_ensemble_2 = predicted_categories_3[1]
                else:
                    predicted_category_ensemble_2 = -5
            if predicted_category_ensemble_1 == -3 or predicted_category_ensemble_1 == -4:
                predicted_category_ensemble_2 = -5

            progressbar_det['value'] += 20

            _checkpoint_category_2 = True

            #####
            # Assigning the third Category (3rd best match)
            ###
            if labels_1[0] == labels_2[1] and predicted_category_ensemble_1 != predicted_categories_1[0] and \
                    predicted_category_ensemble_2 != predicted_categories_1[0] and probabilities_1[0] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_1[0]
            elif labels_1[0] == labels_3[1] and predicted_category_ensemble_1 != predicted_categories_1[0] and \
                    predicted_category_ensemble_2 != predicted_categories_1[0] and probabilities_1[0] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_1[0]
            elif labels_2[0] == labels_1[1] and predicted_category_ensemble_1 != predicted_categories_2[0] and \
                    predicted_category_ensemble_2 != predicted_categories_2[0] and probabilities_2[0] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_2[0]
            elif labels_2[0] == labels_3[1] and predicted_category_ensemble_1 != predicted_categories_2[0] and \
                    predicted_category_ensemble_2 != predicted_categories_2[0] and probabilities_2[0] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_2[0]
            elif labels_3[0] == labels_1[1] and predicted_category_ensemble_1 != predicted_categories_3[0] and \
                    predicted_category_ensemble_2 != predicted_categories_3[0] and probabilities_3[0] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_3[0]
            elif labels_3[0] == labels_2[1] and predicted_category_ensemble_1 != predicted_categories_3[0] and \
                    predicted_category_ensemble_2 != predicted_categories_3[0] and probabilities_3[0] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_3[0]
            elif labels_1[1] == labels_2[1] and predicted_category_ensemble_1 != predicted_categories_1[1] and \
                    predicted_category_ensemble_2 != predicted_categories_1[1] and probabilities_1[1] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_1[1]
            elif labels_1[1] == labels_3[1] and predicted_category_ensemble_1 != predicted_categories_1[1] and \
                    predicted_category_ensemble_2 != predicted_categories_1[1] and probabilities_1[1] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_1[1]
            elif labels_2[1] == labels_3[1] and predicted_category_ensemble_1 != predicted_categories_2[1] and \
                    predicted_category_ensemble_2 != predicted_categories_2[1] and probabilities_2[1] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_2[1]
            elif predicted_category_ensemble_1 != predicted_categories_1[0] and \
                    predicted_category_ensemble_2 != predicted_categories_1[0] and \
                    probabilities_1[0] > probabilities_2[0] and probabilities_1[0] > probabilities_2[1] and \
                    probabilities_1[0] > probabilities_3[0] and probabilities_1[0] > probabilities_3[1] and \
                    probabilities_1[0] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_1[0]
            elif predicted_category_ensemble_1 != predicted_categories_2[0] and \
                 predicted_category_ensemble_2 != predicted_categories_2[0] and \
                 probabilities_2[0] > probabilities_1[0] and probabilities_2[0] > probabilities_1[1] and \
                 probabilities_2[0] > probabilities_3[0] and probabilities_2[0] > probabilities_3[1] and \
                    probabilities_2[0] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_2[0]
            elif predicted_category_ensemble_1 != predicted_categories_3[0] and \
                 predicted_category_ensemble_2 != predicted_categories_3[0] and \
                 probabilities_3[0] > probabilities_1[0] and probabilities_3[0] > probabilities_1[1] and \
                 probabilities_3[0] > probabilities_2[0] and probabilities_3[0] > probabilities_2[1] and \
                    probabilities_3[0] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_3[0]
            elif predicted_category_ensemble_1 != predicted_categories_1[1] and \
                 predicted_category_ensemble_2 != predicted_categories_1[1] and \
                 probabilities_1[1] > probabilities_2[0] and probabilities_1[1] > probabilities_2[1] and \
                 probabilities_1[1] > probabilities_3[0] and probabilities_1[1] > probabilities_3[1] and \
                    probabilities_1[1] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_1[1]
            elif predicted_category_ensemble_1 != predicted_categories_2[1] and \
                 predicted_category_ensemble_2 != predicted_categories_2[1] and \
                 probabilities_2[1] > probabilities_1[0] and probabilities_2[1] > probabilities_1[1] and \
                 probabilities_2[1] > probabilities_3[0] and probabilities_2[1] > probabilities_3[1] and \
                    probabilities_2[1] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_2[1]
            elif predicted_category_ensemble_1 != predicted_categories_3[1] and \
                 predicted_category_ensemble_2 != predicted_categories_3[1] and \
                 probabilities_3[1] > probabilities_1[0] and probabilities_3[1] > probabilities_1[1] and \
                 probabilities_3[1] > probabilities_2[0] and probabilities_3[1] > probabilities_2[1] and \
                    probabilities_3[1] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_3[1]
            elif predicted_category_ensemble_1 != predicted_categories_1[2] and \
                    predicted_category_ensemble_2 != predicted_categories_1[2] and \
                    probabilities_1[2] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_1[2]
            elif predicted_category_ensemble_1 != predicted_categories_2[2] and \
                    predicted_category_ensemble_2 != predicted_categories_2[2] and \
                    probabilities_2[2] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_2[2]
            elif predicted_category_ensemble_1 != predicted_categories_3[2] and \
                    predicted_category_ensemble_2 != predicted_categories_3[2] and \
                    probabilities_3[2] > CONFIDENCE_MIN_3:
                predicted_category_ensemble_3 = predicted_categories_3[2]
            else:
                predicted_category_ensemble_3 = -5

            _checkpoint_category_3 = True

            # Manage order of predictions when insufficient confidence is a factor
            if predicted_category_ensemble_1 == -3 or predicted_category_ensemble_1 == -4:
                predicted_category_ensemble_2 = -5
                predicted_category_ensemble_3 = -5
            elif predicted_category_ensemble_2 == -5 and predicted_category_ensemble_3 != -5 and predicted_category_ensemble_1 != predicted_category_ensemble_3:
                predicted_category_ensemble_2 = predicted_category_ensemble_3
                predicted_category_ensemble_3 = -5

            # Look up the Category Descriptions
            _predicted_category_1 = predicted_category_ensemble_1
            _predicted_category_description_1 = get_category_description_input(predicted_category_ensemble_1)
            _predicted_category_2 = predicted_category_ensemble_2
            _predicted_category_description_2 = get_category_description_input(predicted_category_ensemble_2)
            if _predicted_category_2 == -5:
                _predicted_category_2 = ""
            _predicted_category_3 = predicted_category_ensemble_3
            _predicted_category_description_3 = get_category_description_input(predicted_category_ensemble_3)
            if _predicted_category_3 == -5:
                _predicted_category_3 = ""

            _checkpoint_output_data = True

    except Exception as e:
        log_error(exception=e)

    return


def process_from_input():
    """
    Prepares data entered by user through the GUI and calls the predict_from_input
    function in a separate thread.

    :return:
    """

    global _input_fpath
    global window
    global _user_text
    global _manufacturer_flag
    global _brand_flag
    global _packsize_flag
    global _was_storage_provided
    global lbl_greeting
    global btn_user_input
    global btn_clear_data
    global _input_type
    global _checkpoint_cleaned_data
    global _checkpoint_preprocessed_data
    global _checkpoint_model_1
    global _checkpoint_model_2
    global _checkpoint_model_3
    global _checkpoint_category_1
    global _checkpoint_category_2
    global _checkpoint_category_3
    global _checkpoint_output_data

    _checkpoint_cleaned_data = False
    _checkpoint_preprocessed_data = False
    _checkpoint_model_1 = False
    _checkpoint_model_2 = False
    _checkpoint_model_3 = False
    _checkpoint_category_1 = False
    _checkpoint_category_2 = False
    _checkpoint_category_3 = False
    _checkpoint_output_data = False

    window.update_idletasks()

    frm_finished.grid_remove()
    frm_results.grid_remove()

    _manufacturer_flag = False
    _brand_flag = False
    _packsize_flag = False
    _was_storage_provided = False
    is_storage_error = False
    _input_type = "manual"
    _checkpoint_cleaned_data = False
    _checkpoint_preprocessed_data = False

    # check to make sure the program can locate all the needed model and support files
    has_all_internal_dependencies = check_internal_depencencies()

    try:
        lbl_finished.grid(row=0, column=0, sticky="nsew", padx=5, pady=10)
        progressbar_det.grid(row=0, column=0, sticky="nsew", padx=150, pady=5)
        progressbar_det['value'] = 0
        # progressbar_ind.grid(row=1, column=0, sticky="nsew", padx=325, pady=25)
        progressbar_ind.grid_remove()
        btn_open["state"] = "disabled"
        btn_user_input["state"] = "disabled"
        btn_clear_data["state"] = "disabled"
        frm_progress.grid(row=2, column=0, sticky="nsew")

        if not has_all_internal_dependencies:
            raise Exception("Could not find all internal dependencies")

        # Start moving the indeterminate progress bar.
        # progressbar_ind.start(15)

        description = ent_description.get()
        manufacturer = ent_manufacturer.get()
        brand = ent_brand.get()
        packsize = ent_packsize.get()
        storage = ent_storage.get()

        # Prepare the input data for prediction
        if description != "":
            description = clean_special_characters(description)
            description = clean_description(description)
            description = prune_desc(description)
        if manufacturer != "":
            _manufacturer_flag = True
            manufacturer = clean_special_characters(manufacturer)
            manufacturer = clean_mfr_brand(manufacturer)
            manufacturer = prune_mfr_brand(manufacturer)
        if brand != "":
            _brand_flag = True
            brand = clean_special_characters(brand)
            brand = clean_mfr_brand(brand)
            brand = prune_mfr_brand(brand)
        if packsize != "":
            _packsize_flag = True
            packsize = clean_special_characters(packsize)
            packsize = clean_packsize(packsize)
            packsize = convert_packsize(packsize_text=packsize)
        if storage != "":
            _was_storage_provided = True
            storage = clean_special_characters(storage)
            storage = storage.lower()
            if storage == "shelfstable" or storage == "ref" or storage == "frozen":
                storage = storage
            elif storage == "shelf stable":
                storage = "shelfstable"
            elif storage == "ambient":
                storage = "shelfstable"
            elif storage == "dry":
                storage = "shelfstable"
            elif storage == "cooler":
                storage = "ref"
            elif storage == "fridge":
                storage = "ref"
            elif storage == "refrigerated":
                storage = "ref"
            elif storage == "fzn":
                storage = "frozen"
            elif storage == "frz":
                storage = "frozen"
            elif storage == "frzn":
                storage = "frozen"
            elif storage == "froze":
                storage = "frozen"
            elif storage == "freezer":
                storage = "frozen"
            else:
                is_storage_error = True
                raise Exception("Could not recognize storage terms")

        _checkpoint_cleaned_data = True
        _checkpoint_preprocessed_data = True

        _user_text = description + " " + manufacturer + " " + brand + " " + packsize + " " + storage

        worker_thread = threading.Thread(target=predict_from_input)

        worker_thread.start()

        progressbar_det['value'] = 0

        lbl_greeting["text"] = "To load product information from a file, click the button below."

        schedule_check_input(worker_thread)

    except Exception as e:
        progressbar_ind.stop()
        frm_progress.grid_remove()

        frm_finished.grid(row=0, column=0, sticky="nsew")

        if is_storage_error:
            lbl_finished["text"] = "** Invalid Storage Terms **  Please correct and try again"
        elif not has_all_internal_dependencies:
            lbl_finished["text"] = "** Could not locate all necessary Model / Support Files. Please consult Documentation **"
        else:
            lbl_finished["text"] = "** Unexpected Error **  Please reach out for assistance."
            lbl_greeting["text"] = "To load product information from a file, click the button below."
            log_error(exception=e)

        lbl_finished["foreground"] = "red"

        btn_open["state"] = "normal"
        btn_user_input["state"] = "normal"
        btn_clear_data["state"] = "normal"
        return

    return


def process_from_file():
    """
    Prepares data loaded from .CSV file and calls the predict_from_file
    function in a separate thread.

    :return:
    """

    global _input_fpath
    global btn_user_input
    global btn_clear_data
    global window
    global lbl_greeting
    global _input_type
    global _checkpoint_cleaned_data
    global _checkpoint_preprocessed_data
    global _checkpoint_model_1
    global _checkpoint_model_2
    global _checkpoint_model_3
    global _checkpoint_category_1
    global _checkpoint_category_2
    global _checkpoint_category_3
    global _checkpoint_output_data

    _checkpoint_cleaned_data = False
    _checkpoint_preprocessed_data = False
    _checkpoint_model_1 = False
    _checkpoint_model_2 = False
    _checkpoint_model_3 = False
    _checkpoint_category_1 = False
    _checkpoint_category_2 = False
    _checkpoint_category_3 = False
    _checkpoint_output_data = False

    _input_type = "file"

    if not _did_receive_arguments:
        frm_finished.grid_remove()
        frm_results.grid_remove()

    # check to make sure the program can locate all the needed model and support files
    has_all_internal_dependencies = check_internal_depencencies()

    try:

        """Open a file for editing."""
        filepath = askopenfilename(
            filetypes=[("Excel files", "*.xlsm *.xlsx")]
        )

        if not filepath:
            return

        ent_description.delete(0, 'end')
        ent_manufacturer.delete(0, 'end')
        ent_brand.delete(0, 'end')
        ent_packsize.delete(0, 'end')
        ent_storage.delete(0, 'end')

        _input_fpath = filepath

        lbl_finished.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        progressbar_det.grid(row=0, column=0, sticky="nsew", padx=150, pady=5)
        progressbar_det['value'] = 0
        progressbar_ind.grid(row=1, column=0, sticky="nsew", padx=350, pady=25)
        btn_open["state"] = "disabled"
        btn_user_input["state"] = "disabled"
        btn_clear_data["state"] = "disabled"
        frm_progress.grid(row=2, column=0, sticky="nsew")

        if not has_all_internal_dependencies:
            raise Exception("Could not find all internal dependencies")

        # Start moving the indeterminate progress bar.
        progressbar_ind.start(20)

        worker_thread = threading.Thread(target=predict_from_file)

        # t.daemon = True
        worker_thread.start()

        progressbar_det['value'] = 0

        schedule_check_load(worker_thread)
        # check_if_done_load(worker_thread)

        lbl_greeting["text"] = "To load product information from a file, click the button below."

    except Exception as e:
        # print("Exception occurred in process_from_file function.")
        # print(e)
        progressbar_ind.stop()
        frm_progress.grid_remove()

        frm_finished.grid(row=0, column=0, sticky="nsew")

        if not has_all_internal_dependencies:
            lbl_finished["text"] = "** Could not locate all necessary Model / Support Files. Please consult Documentation **"
        else:
            lbl_finished["text"] = "** Unexpected Error **  Please reach out for assistance."
            lbl_greeting["text"] = "To load product information from a file, click the button below."
            log_error(exception=e)

        lbl_finished["foreground"] = "red"

        btn_open["state"] = "normal"
        btn_user_input["state"] = "normal"
        btn_clear_data["state"] = "normal"
        return

    return


def clear_data():
    """
    Clears the entry fields in the GUI.

    :return:
    """

    global window
    global frm_results
    global frm_finished

    try:

        frm_results.grid_remove()
        frm_finished.grid_remove()

        frm_greeting.grid(row=0, column=0, sticky="nsew")
        frm_load_button.grid(row=1, column=0, sticky="nsew")
        frm_user_input_banner.grid(row=3, column=0, sticky="nsew")

        ent_description.delete(0, 'end')
        ent_manufacturer.delete(0, 'end')
        ent_brand.delete(0, 'end')
        ent_packsize.delete(0, 'end')
        ent_storage.delete(0, 'end')

        return

    except Exception as e:
        log_error(exception=e)


def process_clear_data():
    """
    Starts the clear_data function in a separate thread.

    :return:
    """

    global window

    window.update_idletasks()

    worker_thread = threading.Thread(target=clear_data)
    worker_thread.start()

    schedule_check_clear(worker_thread)

    return


def schedule_check_input(thread):
    """
    Schedules the execution of the check_if_done_input function each
    second.
    """

    global window

    window.after(1000, check_if_done_input, thread)


def check_if_done_input(thread):
    """
    Checks to see if the thread on which the predict_from_input function is
    running and handles actions to be executed upon completion.

    :param thread: The thread on which the predict_from_input function is running
    :return:
    """

    global window
    global frm_results
    global _predicted_category_1
    global _predicted_category_description_1
    global _predicted_category_2
    global _predicted_category_description_2
    global _predicted_category_3
    global _predicted_category_description_3

    try:

        if not thread.is_alive():

            progressbar_ind.stop()
            frm_progress.grid_remove()
            frm_results.grid_remove()

            if _predicted_category_1 == -1:
                frm_finished.grid(row=0, column=0, sticky="nsew", padx=25)
                lbl_finished["text"] = "Description is required. Please enter Description and try again."
                lbl_finished["foreground"] = "red"
            elif _predicted_category_1 == -2:
                frm_finished.grid(row=0, column=0, sticky="nsew", padx=25)
                lbl_finished["text"] = "Not enough recognized terms to proceed."
                lbl_finished["foreground"] = "red"
            elif _predicted_category_1 == -3:
                frm_finished.grid(row=0, column=0, sticky="nsew", padx=25)
                lbl_finished["text"] = "Insufficient confidence. Feel free to review data, make changes, and try again."
                lbl_finished["foreground"] = "red"
            elif _predicted_category_1 == -4:
                frm_finished.grid(row=0, column=0, sticky="nsew", padx=25)
                lbl_finished["text"] = "Insufficient confidence. Please double-check the Food / Non-Food flag and Storage."
                lbl_finished["foreground"] = "red"
            elif _predicted_category_1 == -6:
                frm_finished.grid(row=0, column=0, sticky="nsew", padx=25)
                lbl_finished["text"] = "** Unexpected Error **  Please reach out for assistance."
                lbl_finished["foreground"] = "red"
            else:
                frm_results.grid(row=2, column=0, sticky="nsew", padx=25)
                # If we only have 1 category which met the confidence threshold
                if _predicted_category_2 == "":
                    lbl_results_1["text"] = "Category: " + str(_predicted_category_1) + "  |  " + str(
                        _predicted_category_description_1)
                    lbl_results_2["text"] = ""
                    lbl_results_3["text"] = ""
                # if we have 2 categories which met the confidence threshold
                elif _predicted_category_3 == "":
                    lbl_results_1["text"] = "Category: " + str(_predicted_category_1) + "  |  " + str(
                        _predicted_category_description_1)
                    lbl_results_2["text"] = "Category: " + str(_predicted_category_2) + "  |  " + str(
                        _predicted_category_description_2)
                    lbl_results_3["text"] = ""
                # if we have 3 categories which met the confidence threshold
                else:
                    lbl_results_1["text"] = "Category: " + str(_predicted_category_1) + "  |  " + str(_predicted_category_description_1)
                    lbl_results_2["text"] = "Category: " + str(_predicted_category_2) + "  |  " + str(_predicted_category_description_2)
                    lbl_results_3["text"] = "Category: " + str(_predicted_category_3) + "  |  " + str(_predicted_category_description_3)

            btn_open["state"] = "normal"
            btn_user_input["state"] = "normal"
            btn_clear_data["state"] = "normal"
        else:
            # Otherwise check again after one second.
            schedule_check_input(thread)

    except Exception as e:
        log_error(exception=e)


def schedule_check_load(thread):
    """
    Schedules the execution of the check_if_done_load function each
    second.
    """

    global window

    window.after(1000, check_if_done_load, thread)


def check_if_done_load(thread):
    """
    Checks to see if the thread on which the predict_from_file function is
    running and handles actions to be executed upon completion.

    :param thread: The thread on which the predict_from_file function is running
    :return:
    """

    global window
    global _is_invalid_file
    global _is_missing_headers
    global _is_more_than_2000
    global _output_saved_successfully
    global lbl_greeting

    try:

        if not thread.is_alive():

            progressbar_ind.stop()
            frm_progress.grid_remove()

            if _is_invalid_file:
                lbl_finished["text"] = "** Unexpected File **  Please check file or reach out for assistance"
                lbl_finished["foreground"] = "red"
                lbl_greeting["text"] = "To load product information from a file, click the button below."
            elif _is_missing_headers and not _is_more_than_2000:
                lbl_finished["text"] = "** Missing Headers **  File requires specific headers. Please consult Documentation."
                lbl_finished["foreground"] = "red"
                lbl_greeting["text"] = "To load product information from a file, click the button below."
            elif _is_more_than_2000:
                lbl_finished["text"] = "** More than 2000 Products **  Files should contain 2000 or less."
                lbl_finished["foreground"] = "red"
                lbl_greeting["text"] = "To load product information from a file, click the button below."
            elif _output_saved_successfully:
                lbl_finished["text"] = "Processing completed!  Output file successfully saved."
                lbl_finished["foreground"] = "green"
                lbl_greeting["text"] = "To load product information from a file, click the button below."
            else:
                lbl_finished["text"] = "** Unexpected Error **  Please reach out for assistance."
                lbl_finished["foreground"] = "red"
                lbl_greeting["text"] = "To load product information from a file, click the button below."

            frm_finished.grid(row=2, column=0, sticky="nsew")
            btn_open["state"] = "normal"
            btn_user_input["state"] = "normal"
            btn_clear_data["state"] = "normal"

        else:
            # Otherwise check again after one second
            schedule_check_load(thread)

    except Exception as e:
        log_error(exception=e)


def schedule_check_clear(thread):
    """
    Schedules the execution of the check_if_done_clear function each
    second.
    """

    global window

    window.after(1000, check_if_done_clear, thread)


def check_if_done_clear(thread):
    """
    Checks to see if the thread on which the clear_data function is
    running and handles graceful return to main loop.

    :param thread: The thread on which the clear_data function is running
    :return:
    """

    global window

    window.update_idletasks()

    if not thread.is_alive():
        return

    else:
        # Otherwise check again after one second.
        schedule_check_clear(thread)


def get_category_description_load(category_list):
    """
    Checks the category ids from a list and returns the corresponding
    category descriptions.

    Keyword arguments:
    category_list (list) -- List of the predicted categories

    Return:
    desc_list (list) -- The descriptions for the predicted categories
    """

    try:

        # load the category description data
        mapping_list = []
        mapping_filename = MODEL_1_CATEGORY_DESCRIPTIONS_MAPPING
        with open(mapping_filename, 'r') as mapping_data:
            for line in csv.DictReader(mapping_data):
                mapping_list.append(line)
        mapping_decoder = {}

        for i in range(len(mapping_list)):
            mapping_decoder[mapping_list[i]['category']] = mapping_list[i]['category_description']

        desc_list = []

        for item in category_list:
            if str(item) == '-1':
                desc = 'n/a'
            elif str(item) == '-2':
                desc = 'n/a'
            elif str(item) == '-3':
                desc = 'n/a'
            elif str(item) == '-4':
                desc = 'n/a'
            elif str(item) == '-5':
                desc = 'n/a'
            else:
                desc = mapping_decoder[str(item)]

            desc_list.append(desc)

        return desc_list

    except Exception as e:
        log_error(exception=e)


def get_category_description_input(category_id):
    """
    Checks the category id and returns the corresponding category description.

    Keyword arguments:
    category_id (int) -- The predicted category id

    Return:
    desc (str) -- The description for the predicted category
    """

    try:

        # load the catgegory description data
        mapping_list = []
        mapping_filename = MODEL_1_CATEGORY_DESCRIPTIONS_MAPPING
        with open(mapping_filename, 'r') as mapping_data:
            for line in csv.DictReader(mapping_data):
                mapping_list.append(line)
        mapping_decoder = {}

        for i in range(len(mapping_list)):
            mapping_decoder[mapping_list[i]['category']] = mapping_list[i]['category_description']

        if str(category_id) == 'Unable to make confident determination.':
            desc = 'n/a'
        elif category_id == -2:
            desc = 'n/a'
        elif category_id == -3:
            desc = 'Unable to make confident determination'
        elif category_id == -4:
            desc = 'n/a'
        elif category_id == -5:
            desc = 'n/a'
        else:
            desc = mapping_decoder[str(category_id)]

        return desc

    except Exception as e:
        log_error(exception=e)


def check_internal_depencencies():

    global MODEL_1_CATEGORY_DESCRIPTIONS_MAPPING
    global MODEL_1_LABEL_MAPPING
    global MODEL_2_CATEGORY_DESCRIPTIONS_MAPPING
    global MODEL_2_LABEL_MAPPING
    global MODEL_3_CATEGORY_DESCRIPTIONS_MAPPING
    global MODEL_3_LABEL_MAPPING
    global MODEL_1_PATH
    global VOCAB_1_PATH
    global MODEL_2_PATH
    global VOCAB_2_PATH
    global MODEL_3_PATH
    global VOCAB_3_PATH

    try:

        path_list = [MODEL_1_CATEGORY_DESCRIPTIONS_MAPPING, MODEL_1_LABEL_MAPPING, MODEL_2_CATEGORY_DESCRIPTIONS_MAPPING,
                     MODEL_2_LABEL_MAPPING, MODEL_3_CATEGORY_DESCRIPTIONS_MAPPING, MODEL_3_LABEL_MAPPING,
                     MODEL_1_PATH, VOCAB_1_PATH, MODEL_2_PATH, VOCAB_2_PATH, MODEL_3_PATH, VOCAB_3_PATH]

        for item in path_list:
            if not os.path.exists(item):
                return False

        return True

    except Exception as e:
        log_error(exception=e)


def main():
    """
    The main function of the script.

    :return: None
    """

    global btn_open
    global btn_user_input
    global btn_clear_data
    global window
    global _did_receive_arguments

    try:

        # If arguments received, run the classifier in headless mode
        if _did_receive_arguments:
            predict_from_file()

        # If no arguments received, open the GUI and run the main window loop
        else:

            btn_open = tk.Button(frm_load_button, text="Load File", border=4, command=process_from_file)
            btn_open.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

            btn_user_input = tk.Button(frm_user_input_button, text="Submit Data", border=4, command=process_from_input)
            btn_user_input.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

            btn_clear_data = tk.Button(frm_user_input_button, text="Clear", border=4, command=process_clear_data)
            btn_clear_data.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

            frm_greeting.grid(row=0, column=0, sticky="nsew")
            frm_load_button.grid(row=1, column=0, sticky="nsew")

            frm_user_input_banner.grid(row=3, column=0, sticky="nsew")
            frm_user_input.grid(row=5, column=0, sticky="nsew")
            frm_user_input_button.grid(row=6, column=0, sticky="nsew")

            # make sure the window is visible and bring it to the front
            window.deiconify()
            window.lift()

            window.mainloop()

        try:
            window.quit()
            window.destroy()
            sys.exit()
        except:
            sys.exit()

    except Exception as e:
        log_error(exception=e)


if __name__ == "__main__":
    main()

try:
    window.quit()
    window.destroy()
    sys.exit()
except:
    sys.exit()
