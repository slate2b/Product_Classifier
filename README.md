# Product Classifier

The Product Classifier application is designed to analyze product data and determine the correct category 
within a given taxonomy.  The models used in the application utilize elements from the Transformer architecture, 
but the scope is more focused.  Instead of using a Decoder Only or an Encoder/Decoder, I chose to build models 
around a pair of relatively small Encoder-Only modules.  This smaller size means that they cannot be used to 
mimic human behavior across a wide range of applications, however, when trained on data for a specific use case, 
these models can outperform the latest LLMs--even after prompt-engineering and fine-tuning.  This is 
especially valuable when dealing with a business-critical task that demands high accuracy, and these models
routinely achieve accuracy scores higher than 95% in the product classification task even in Taxonomies with 
2000+ categories.  

<img width="851" height="710" alt="image" src="https://github.com/user-attachments/assets/e2c43a5f-1208-4a86-8d3c-1d5131ec37b4" />


This project centers around the Product Classifier application, which presents the user with a simple GUI capable
of accepting product data through file uploads or through direct manual entry.  It also includes a collection of 
scripts used for cleansing and preprocessing product data for classification, along with a model trainer.  

The models used in the application are built using a custom architecture which utilizes an Embedding layer, a
Positional Encoding layer, a Multihead Attention layer, Dual Multilayer Encoders, and a final Classification layer.

Each of the Dual Encoders follows design elements similar to Encoder-Only Transformer models like BERT.  Although the
encoders are probably the most notable components in architecture, it is the collaboration between the Embedding,
Positional Encoding, Multihead Attention, and the Dual Encoders which makes the classification task successful.

# Text Preprocessing

The Product Classifier models are designed for text-based input, but unlike many popular models which are built to
make sense of natural language (statements, questions, common forms of human speech), these models are designed to make
sense of product data in as it is commonly found on a product specification sheet.  This includes the description, brand
name, manufacturer name, pack size, country of origin, and storage data.  Like other text-based models, the Product
Classifier models require the text to be preprocessed into forms that help them function effectively.  The classifier
utilizes a data cleaner and a preprocessor which are custom designed for these models.  Once the text is preprocessed,
the classifier program tokenizes the text (converts each text term to a numeric value) which is the last step before
feeding the data to the models.

# Model Processing

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
