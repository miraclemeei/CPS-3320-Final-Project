# %%
# import the libraries
import pandas as pd
import numpy as np
import sqlite3
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Loading the data from the dataset
def readData():
    content = sqlite3.connect(r"D:\Documents\Data\archive\database.sqlite")
    data = pd.read_sql_query(""" SELECT * FROM Reviews""", content)
    print(data.shape)
    return data


def DataCleansing(data):
    # DataCleansing
    # Cleaning of the duplicated including checking the null values
    df = data.drop_duplicates(subset=['UserId', 'ProfileName', 'Time', 'Text'], keep='first', inplace=False)
    df.shape

    # Start of Building the dataset and for data preparation
    data = df[['Score', 'Text']]  # Slicing the array to omit other information that is essential for analysis
    data['sentiments'] = data.Score.apply(
        lambda x: 0 if x in [1, 2] else 1)  # using lambda to omit a score of 3, in other words, the neutral
    data.head()
    data.drop(data[data['Score'] == 3].index, inplace=True)
    data.head()
    return data


# %%This part including the hole data mining with Cross Validation , Deep Learning, Tokenization.
def dataMining(data, epoch, batch):
    # Cross Validation
    split = round(len(data) * 0.8)  # train : test = 8: 2
    train_reviews = data['Text'][:split]
    train_label = data['sentiments'][:split]
    test_reviews = data['Text'][split:]
    test_label = data['sentiments'][split:]

    # %%

    """
    Here start the Tokenization
    tokenization，also known as word segmentation
    """

    tokenizer = Tokenizer(oov_token="<OOV>")  # oov stands for out of vocabulary

    # %%

    # Cross Validation //Storing the Records in an array as the network need array like input
    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []
    for row in train_reviews:
        training_sentences.append(str(row))
    for row in train_label:
        training_labels.append(str(row))
    for row in test_reviews:
        testing_sentences.append(str(row))
    for row in test_label:
        testing_labels.append(str(row))

    # %%

    """
    num_words is the dictionary size 
    """

    # <OOV> is Out of Vocabulary total dict length will be 80000
    tokenizer = Tokenizer(num_words=80000, oov_token="<OOV>")
    tokenizer.fit_on_texts(training_sentences)  # token to the array
    word_index = tokenizer.word_index  # word_index is a dict later methods will be referred to this dict

    # %%

    # padding the length in order to make it a unity form.
    sequences = tokenizer.texts_to_sequences(training_sentences)  # change the sequence to a set of the
    padded = pad_sequences(sequences, maxlen=100, truncating='post', padding='post')
    testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sentences, maxlen=100, truncating='post', padding='post')

    # %%

    # 1, Word Embedding (same meaning reduction of dimension )2. Golbal average layer, prevent over fitting 3,dense layer 16 units relu 4, output layer
    model = keras.Sequential([keras.layers.Embedding(80000, 9, input_length=100),  # 9 - deminsion vector
                              keras.layers.GlobalAvgPool1D(),
                              # for getting the average of the word with the same meaning
                              keras.layers.Dense(6, activation='relu'),  # activation fucnction
                              keras.layers.Dense(1, activation='sigmoid')])  # smooth function

    # %%

    # adam as an gradient descent binary——cross as a loss function, accuracy to be the testing point
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # %%

    # change the str in dict to int
    training_labels_final = np.array(training_labels).astype(int)
    testing_labels_final = np.array(testing_labels).astype(int)

    # %%

    history = model.fit(padded, training_labels_final, batch_size=batch, epochs=epoch,
                        validation_data=(testing_padded, testing_labels_final))

    # This part plot the traning and testing accuracy
    import matplotlib.pyplot as plt
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title('Accuracy for Deep Learning', c='r')
    plt.figure()


# %%
def main():
    try:  # user input for hyperparameters epochs and batches.
        print("Input Valid Integer for epochs")
        epoch = int(input())
        print("Input Valid Integer for batches")
        batch = int(input())
    except:  # if not integer then throw the exception
        print("You Must Input Valid Integer for epochs and batches")

    data = readData()
    data = DataCleansing(data)
    dataMining(data, epoch, batch)

if __name__ == "__main__":
    main()
