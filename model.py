from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, MaxPooling1D, Conv1D
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random, sys, io, re, csv, pickle, time, os
from examplebase import ModelData

songNumber = 100
maxSongLen = 500 #Not used in the program
maxlen = 60 # Max length of input example
step = 2 # Step / stride of the considered window

startTime = time.time()

# Determine if the data model with the current settings exists
fileName = "data/modelData_%d_%d_%d.p" % (songNumber, maxlen, step)
if os.path.exists(fileName):
    # Read in the Example Data base, so that we don't have to restructure one-hot encodings
    with open("modelData.p", "rb") as pFile:
        md = pickle.load(pFile)
else:
    md = ModelData()

## Get the data
if md.hasData():
    print("Data exists, loading from file...")
    words = md.getVocab()
    textAsWords = md.getLyrics()
    words_indices = md.getWords2Indices()
    indices_words = md.getIndices2Words()
    inputs = md.getInputs()
    targets = md.getTargets()
else:
    print("Data doesn't exist, processing from file...")
    with open("finaltokens.csv", "r") as csvFile:
        r = csv.reader(csvFile, delimiter=',') 
        words = [] #List of words in the corpus

        ## Get the words from the text
        for i, line in enumerate(r):
            #print(i)
            words.extend(line)

        # Remove duplicates and sort the vocabulary
        textAsWords = words
        words = sorted(list(set(words))) 

        # Store the vocab and lyrics to the model data object
        md.setVocab(words)
        md.setLyrics(textAsWords)
        
    ## Create mappings between words and their indices in the vocab
    words_indices = dict((c, i) for i, c in enumerate(words))
    indices_words = dict((i, c) for i, c in enumerate(words))

    # Create initial values for the inputs and targets
    inputs = np.zeros((0, maxlen, len(words)), dtype=np.bool) #Inputs into the model
    targets = np.zeros((0, len(words)), dtype=np.bool) #Target output for each input example

    with open("finaltokens.csv", "r") as csvFile2:
        
        # Read in each song
        r = csv.reader(csvFile2, delimiter=',')
        
        for j, line in enumerate(r):
            
            if j < songNumber:

                excerpts = [] # maxLen size pieces of text
                nextWords = [] # all text beyond the excerpt of the particular song
                # Iterate through the song up to its length with a given step size
                for i in range(0, len(line) - maxlen, step):
                    excerpts.append(line[i: i + maxlen]) #Grab excerpt
                    nextWords.append(line[i + maxlen]) #Grab words that follow the excerpt

                ## Vectorize the training examples (as a 3-D Tensor)
                x = np.zeros((len(excerpts), maxlen, len(words)), dtype=np.bool) ## create space for input data
                y = np.zeros((len(excerpts), len(words)), dtype=np.bool) ## create space for targets
                for i, excerpt in enumerate(excerpts): ## Iterate through the excerpts
                    for t, word in enumerate(excerpt): ## Iterate through the words in the excerpt
                        x[i, t, words_indices[word]] = 1 ## Set index to 1 for word (one-hot encoding)
                    y[i, words_indices[nextWords[i]]] = 1 ## Set index to 1 for word (one-hot encoding)

                inputs = np.concatenate((inputs, x)) # Add the new x input examples to the inputs list
                targets = np.concatenate((targets, y)) # Add the new y targets to the targets list

    # Store the inputs and targets to the model data object               
    md.setInputs(inputs)
    md.setTargets(targets)
    md.setLoaded() # Tell the model object that it has been loaded with data

    # Repickle the data object
    with open(fileName, "wb") as pFile:
        pickle.dump(md, pFile, protocol=pickle.HIGHEST_PROTOCOL)

# Break the data into training, validation, and testing sets
examples = inputs.shape[0]
split1 = int(examples*.6)
split2 = int(examples*.8)

x_train = inputs[:split1] 
y_train = targets[:split1]

x_val = inputs[split1:split2]
y_val = targets[split1:split2]

x_test = inputs[split2:]
y_test = targets[split2:]

# Print the time it took to run the above
print("Load time:", round(time.time() - startTime, 2), "secs")

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Conv1D(32,5,input_shape=(maxlen, len(words))))
model.add(MaxPooling1D(2))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(len(words), activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(textAsWords) - maxlen - 1) #Grabs random seed from entire corpus
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = []
        excerpt = textAsWords[start_index: start_index + maxlen]
        generated += excerpt
        print('----- Generating with seed: "' + " ".join(excerpt) + '"')
        print("-"*40)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(excerpt):
                x_pred[0, t, words_indices[word]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_words[next_index]

            excerpt = excerpt[1:]
            excerpt.append(next_word)
        print(" ".join(excerpt))
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x_train, y_train,
          batch_size=128,
          validation_data=(x_val, y_val),
          epochs=2)#,
          #callbacks=[print_callback])

score, acc = model.evaluate(x_test, y_test)

on_epoch_end(60, None)

print("Accuracy:", acc)
