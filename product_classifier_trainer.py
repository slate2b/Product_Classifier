"""
PRODUCT CLASSIFIER TRAINER

Developed By: Thomas Vaughn
Version: 1.1.4
Last Update Date: 7/10/2025

This training script is designed to create and train models using a custom architecture which utilizes an Embedding
layer, a Positional Encoding layer, a Multihead Attention layer, Dual Multilayer Encoders, and a final Classification
layer.

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


import torch
import math
from data_preprocessor_for_trainer import prepare_dataset
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
import time
from torchtext.data.functional import to_map_style_dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

source_filepath = './source_data/Baseline_Dataset.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processed_train_fpath = 'source_data/Train_Processed_Dataset.csv'
processed_validation_fpath = 'source_data/Validation_Processed_Dataset.csv'
processed_test_fpath = 'source_data/Test_Processed_Dataset.csv'

_model_save_path = './product_classifier_1.pt'
_optimizer_save_path = './product_classifier_1_optimizer.pt'
_vocab_save_path = './product_classifier_1_vocab.pt'
src_vocab_size = 0

_dim_term = 512  # dimensions of the model (number of values used to represent single term)
_d_ff = _dim_term * 4  # dimensions used in the feed forward network - the 4 is for q, k, v, o in MultiheadAttention

_num_layers_1 = 6  # number of primary encoder layers the model uses to process the input
_num_heads_1 = 16  # number of perspectives from which the model processes the input

_num_layers_2 = 6  # number of primary encoder layers the model uses to process the input
_num_heads_2 = 16  # number of perspectives from which the model processes the input

_max_seq_length = 26  # max number of terms the model can manage per product record
_dropout = 0.1
_fc_dropout = 0.3
_epochs = 25
_learning_rate = .000002
_batch_size = 128
_num_batches = 0
_num_labels = 2029  # number of categories in the data
_criterion = torch.nn.CrossEntropyLoss()
_positional_log_base = 10000.0

train_iter = []
validation_iter = []
test_iter = []
text_pipeline = "placeholder"
label_pipeline = "placeholder"
train_dataloader = "placeholder"
valid_dataloader = "placeholder"
test_dataloader = "placeholder"


def yield_tokens(tknzr, data_iter):
    """
    Tokenizes text using the given tokenizer.

    yield: The tokenized text

    :param tknzr: (TorchText Tokenizer): The tokenizer to be used
    :param data_iter: (list): An iterable list containing the data to be tokenized
    """
    for _, text in data_iter:
        yield tknzr(text)


def build_iter(data_list):
    """
    Builds a list of tuples containing the labels and text from a data list.

    :param data_list: (list): A list version of the dataset split
    :return iterable_list (list): list of tuples containing the labels and text from the given df
    """

    iterable_list = []

    for i in range(len(data_list.index)):
        label = data_list['category'].iloc[i]
        text = data_list['description'].iloc[i]
        iter_tuple = (label, text)
        iterable_list.append(iter_tuple)

    return iterable_list


def init_iterators(trn_list, val_list, tst_list):
    """
    Creates iterable lists (lists of tuples) for the train,
    validation, and test splits.

    :param trn_list: (list): Data list for the training split
    :param val_list: (list): Data list for the validation split
    :param tst_list: (list): Data list for the test split
    :return None
    """

    global train_iter
    global validation_iter
    global test_iter

    # Create iterable lists for each dataset split
    train_iter = build_iter(trn_list)
    validation_iter = build_iter(val_list)
    test_iter = build_iter(tst_list)

    return


def build_vocab(tknizer, data_iter):
    """
    Builds a custom vocabulary based on the given tokenizer and text data.

    :param tknizer: (TorchText Tokenizer): The tokenizer to be used
    :param data_iter: (list of tuples): An iterable list containing the text data
    :return vocab (TorchText Vocab)
    """

    # Build the vocabulary using the tokenizer defined in global variables
    # vocab = build_vocab_from_iterator(iterator=yield_tokens(tknizer, data_iter), min_freq=10, specials=["<unk>"])
    vocab = build_vocab_from_iterator(iterator=yield_tokens(tknizer, data_iter), max_tokens=25000, specials=["<unk>"])
    # vocab = build_vocab_from_iterator(iterator=yield_tokens(tknizer, data_iter), max_tokens=27000, specials=["<unk>"])
    # vocab = build_vocab_from_iterator(iterator=yield_tokens(tknizer, data_iter), max_tokens=30000, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab


def init_pipelines(vcb, tknzr):
    """
    Creates text and label pipelines to be used with the DataLoaders.

    :param vcb: (TorchText Vocab): The vocabulary to be used
    :param tknzr: (TorchText Tokenizer): The tokenizer to be used
    :return None
    """
    # Create pipelines for text processing
    global text_pipeline
    global label_pipeline

    text_pipeline = lambda x: vcb(tknzr(x))
    label_pipeline = lambda x: int(x)

    return


def collate_batch(batch):
    """
    Collate function to be used with Torch DataLoaders.

    The text_list which this function builds is a list of tensors, each tensor containing an integer
    representation of a word (token). The number of tensors, len(text_list), matches the batch size.

    The function also uses pad_sequence from torch rnn utils to pad each record to the length
    (number of tokens) of the largest record in the batch.  This allows us to capture token relationships
    in the embedding layer for a bi-directional lstm model.

    :param batch: The batch to be collated
    :return label_list, text_list, offsets (Torch Tensors): The batched labels, text, and offsets
    """

    global text_pipeline
    global label_pipeline
    global _batch_size

    global text_pipeline
    global label_pipeline
    global _batch_size
    global _max_seq_length

    # label_list, text_list, length_list = [], [], []
    label_list, text_list = [], []

    # Loop through the products in the batch
    for _label, _text in batch:
        # Run the label for a single product through the label_pipeline (cast to int) and append the int label to the label_list
        label_list.append(label_pipeline(_label))

        # Run the text for a single product through the text_pipeline (tokenize it) and convert the text to a tensor of those tokens
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)

        # Append the tokenized text tensor for the single product to the text_list
        text_list.append(processed_text)

    text_list[0] = nn.ConstantPad1d((0, _max_seq_length - text_list[0].shape[0]), 0)(text_list[0])

    # Convert the label_list to a tensor
    labels = torch.tensor(label_list, dtype=torch.int64)

    # Pad each product to the max number of tokens in the batch and convert the text_list to a tensor
    text = pad_sequence(text_list, batch_first=True)

    # return labels.to(device), text.to(device), text_lengths.to(device)
    return labels.to(device), text.to(device)


def init_dataloaders():
    """
    Creates Torch DataLoaders for each dataset split (train, valid, and test).

    :return None
    """
    global train_dataloader
    global valid_dataloader
    global test_dataloader
    global train_iter
    global validation_iter
    global test_iter
    global _batch_size

    # Convert the dataset splits to map style datasets
    train_dataset = to_map_style_dataset(train_iter)
    validation_dataset = to_map_style_dataset(validation_iter)
    test_dataset = to_map_style_dataset(test_iter)

    # Create DataLoaders to manage batches to send to the model
    train_dataloader = DataLoader(
        train_dataset, batch_size=_batch_size, shuffle=False, collate_fn=collate_batch
    )
    valid_dataloader = DataLoader(
        validation_dataset, batch_size=_batch_size, shuffle=False, collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=_batch_size, shuffle=False, collate_fn=collate_batch
    )

    return


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

        global _dim_term
        global _d_ff
        global _num_layers_1
        global _num_heads_1
        global _num_layers_2
        global _num_heads_2

        self.encoder_embedding = nn.Embedding(src_vocab_size, _dim_term)
        self.positional_encoding = PositionalEncoding(_dim_term, max_seq_length)
        self.dropout1 = nn.Dropout(dropout)
        self.encoder_1 = BasicEncoder(_dim_term, _num_heads_1, _num_layers_1, _d_ff, dropout, num_labels)
        self.encoder_2 = BasicEncoder(_dim_term, _num_heads_2, _num_layers_2, _d_ff, dropout, num_labels)
        self.norm1 = nn.LayerNorm(_dim_term)
        self.fc1 = nn.Linear(_dim_term, num_labels)

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

        return output


def checkpoint(model, optimizer, filename):
    model_fp = _model_save_path + filename
    optimizer_fp = _optimizer_save_path + filename

    torch.save(model.state_dict(), model_fp)
    torch.save(optimizer.state_dict(), optimizer_fp)


def resume(model, optimizer, filename):
    model_fp = _model_save_path + filename
    optimizer_fp = _optimizer_save_path + filename

    model.load_state_dict(torch.load(model_fp))
    optimizer.load_state_dict(torch.load(optimizer_fp))

    return model, optimizer


def train(model, loader, optimizer, epch):
    """
    Perform a single epoch of training.

    :param model: (nn.Module): The model to be trained
    :param loader: (Torch DataLoader): The DataLoader to generate training batches
    :param optimizer: (Torch Optimizer): The Optimizer to use for training
    :param epch: (int): The current epoch
    :return:
    """

    global _num_batches

    model.train()  # set model to training mode
    total_acc, total_count, loss = 0, 0, 0
    log_interval = _num_batches
    start_time = time.time()
    progress_start_time = time.time()

    for idx, (label, text) in enumerate(loader):

        optimizer.zero_grad()
        predicted_label = model(text)

        loss = _criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        accu_val = total_acc / total_count
        loss_val = loss.item()

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print("\n")
            print("-" * 89)
            print(
                "| epoch {:3d} |  {:5d}/{:5d} batches "
                "| train accuracy {:8.3f} | train loss {:14.4f}".format(
                    epch, idx+1, len(loader), total_acc / total_count, loss.item()
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

    return accu_val, loss_val


def evaluate(model, loader):
    """
    Uses the given dataset split to evaluate the trained model.

    :param model: (nn.Module): The model to be evaluated
    :param loader: (Torch DataLoader): The DataLoader to generate training batches
    :return avg_acc, loss
    """

    model.eval()  # set model to evaluation mode
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(loader):
            predicted_label = model(text)
            loss = _criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            avg_acc = total_acc / total_count
    return avg_acc, loss


def train_model(model, optimizer):
    """
    Manages training, evaluating, and saving the model.

    :param model: (nn.Module): The model to be trained
    :param optimizer: (Torch Optimizer): The optimizer to be used for training
    :return None
    """

    global _model_save_path
    global _optimizer_save_path
    global train_dataloader
    global valid_dataloader
    global test_dataloader

    validation_acc_history = []
    validation_loss_history = []
    train_acc_history = []
    train_loss_history = []

    is_finished = False

    best_acc = 0.0
    best_loss = 100.0

    start_epoch = 1
    if start_epoch > 1:
        resume_epoch = start_epoch
        resume(model, optimizer, f"epoch_{resume_epoch}.pt")

    # Train and run a validation eval for each epoch
    for epoch in range(1, _epochs + 1):

        if not is_finished:

            epoch_start_time = time.time()
            train_stats = train(model=model, loader=train_dataloader, optimizer=optimizer, epch=epoch)

            train_epoch_accuracy = train_stats[0]
            train_epoch_loss = train_stats[1]

            train_acc_history.append(train_epoch_accuracy)
            train_loss_history.append(train_epoch_loss)

            if not is_finished:

                accu_val, loss_val = evaluate(model, valid_dataloader)
                print("-" * 89)
                print(
                    "| end of epoch {:3d} | time: {:6.2f}s | "
                    "valid accuracy {:8.3f} | valid loss {:14.4f}".format(
                        epoch, time.time() - epoch_start_time, accu_val, loss_val
                    )
                )
                print("-" * 89)

                epoch_accuracy = accu_val
                epoch_loss = loss_val.item()

                # Check to see if model failed to improve significantly since last epoch
                is_acc_less = False
                is_acc_not_much_better = False
                is_loss_significantly_worse = False
                is_loss_small_enough = False

                if epoch_accuracy <= best_acc:
                    is_acc_less = True

                if epoch_accuracy - best_acc < 0.002:
                    is_acc_not_much_better = True

                if epoch > 4 and epoch_loss - best_loss > 0.05:
                    is_loss_significantly_worse = True

                if (epoch > 1 and (is_acc_less and is_loss_significantly_worse) or
                        (not is_acc_less and is_acc_not_much_better and is_loss_significantly_worse)):
                    print("\nEarly stopped training at epoch %d" % epoch)

                    is_finished = True

                    break  # exit the current loop and return to the is_finished condition check

                if epoch_accuracy > best_acc:
                    best_acc = epoch_accuracy
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                if best_acc == epoch_accuracy or best_loss == epoch_loss:
                    checkpoint(model, optimizer, f"_epoch_{epoch}.pt")

                if best_acc > 0.993 and best_loss < 0.005:
                    print("\nEarly stopped training at epoch %d because current epoch stats have met the threshold" % epoch)
                    is_finished = True
                    break  # exit the current loop and return to the is_finished condition check

                validation_acc_history.append(epoch_accuracy)
                validation_loss_history.append(epoch_loss)

    # Perform an evaluation based on the test dataset split
    print("\nChecking the results of test dataset...\n")

    # Evaluate the model
    accu_test, loss_test = evaluate(model, test_dataloader)
    print("test accuracy {:8.3f} | test loss {:8.4f}".format(accu_test, loss_test))

    # Print model's state_dict
    print("\n")
    print("\nModel's state_dict:\n")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("\nSaving model...\n")

    torch.save(model.state_dict(), _model_save_path)
    torch.save(optimizer.state_dict(), _optimizer_save_path)

    return


def get_num_params(model):
    num_params = 0
    for i in list(model.parameters()):
        num_neurons = 1
        for j in list(i.size()):
            num_neurons = num_neurons * j
        num_params += num_neurons
    return num_params


def main():
    """
    The main function of the script.

    :return: None
    """

    # Define global variables
    global train_iter
    global validation_iter
    global test_iter
    global _num_batches
    global _batch_size
    global _vocab_save_path
    global _num_labels
    global src_vocab_size

    print("\nPreparing dataset for training...\n")

    train_data, validation_data, test_data = prepare_dataset(source_fpath=source_filepath)

    # Saving data to file for future reference
    train_data.to_csv(processed_train_fpath, sep=',', encoding='utf-8', index=False)
    validation_data.to_csv(processed_validation_fpath, sep=',', encoding='utf-8', index=False)
    test_data.to_csv(processed_test_fpath, sep=',', encoding='utf-8', index=False)
    train_data = pd.read_csv(processed_train_fpath, encoding='utf-8', dtype=str)
    validation_data = pd.read_csv(processed_validation_fpath, encoding='utf-8', dtype=str)
    test_data = pd.read_csv(processed_test_fpath, encoding='utf-8', dtype=str)

    # Calulate number of batches based on size of training dataset
    _num_batches = len(train_data) / _batch_size
    if isinstance(_num_batches, float):
        _num_batches = len(train_data) // _batch_size
    else:
        _num_batches = (len(train_data) // _batch_size) - 1

    # Initialize iterable lists for each split
    init_iterators(train_data, validation_data, test_data)

    # Get a basic english tokenizer to use for custom vocabulary
    tokenizer = get_tokenizer("basic_english")

    # Build the vocabulary for the model
    vocabulary = build_vocab(tokenizer, train_iter)

    # Define the vocab size based on the vocab object
    vocab_size = len(vocabulary)

    # Save the vocabulary
    torch.save(vocabulary, _vocab_save_path)

    # Initialize the text and label pipelines for the dataloaders
    init_pipelines(vocabulary, tokenizer)

    # Initialize the DataLoaders for training
    init_dataloaders()

    # Define the model and optimizer
    model = MultiEncoderPosEncMean(vocab_size, _max_seq_length, _dropout, num_labels=_num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_learning_rate, betas=(0.9, 0.98), eps=1e-9)

    print("\nTraining the model...\n")

    # Train the model
    train_model(model, optimizer)

    # Run the trained model against the test set
    model.eval()

    # Print the number of parameters
    print("\nNumber of parameters in model:")
    print(get_num_params(model))

    print("\n******************\n"
          "Training complete.\n"
          "******************\n")

    exit()


if __name__ == "__main__":
    main()
