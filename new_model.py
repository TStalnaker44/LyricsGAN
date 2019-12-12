from __future__ import print_function
from keras import backend as K
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM, Bidirectional, Embedding
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from gensim.models import KeyedVectors
import numpy as np
import random, io, csv, nltk

class Generator():

    def __init__(self):
        """Initialize the generator"""
        self.initializeParams()
        self.loadData()
        self.createModel()

    def initializeParams(self):
        """Initialize the parameters of the model"""
        self._epochs = 500
        self._batchSize = 64
        self._embedDepth = 10
        self._songNumber = 5000
        self._ENDINGS_CSV = "endings.csv"

    def loadData(self):
        # Read in the token data from csv and join them as strings
        with open("finaltokens.csv", "r") as csvFile:
            r = csv.reader(csvFile, delimiter=',')
            self._data = ""
            self._terms = []
            for i, line in enumerate(r):
                if i < self._songNumber:
                    self._data += " ".join(line) + "\n"
                    self._terms.extend(line)
            self._terms = sorted(list(set(self._terms)))

        # Tokenize the data (again)
        #Abby's tokenization was perfect but keras needed to be happy
        self._tokenizer = Tokenizer()
        self._tokenizer.fit_on_texts([self._data])
        self._encoded = self._tokenizer.texts_to_sequences([self._data])[0]

        # Determine the size of the vocabulary ie how many tokens
        self._vocabLen = len(self._tokenizer.word_index) + 1

        # Create word sequences from words
        sequences = np.array([self._encoded[i-1:i+1] for i in range(1, len(self._encoded))])

        # Split the sequences into input and targets
        x, y = sequences[:,0],sequences[:,1]

        # One-hot encode the targets using Keras' built in to_categorical
        y = to_categorical(y, num_classes=self._vocabLen)

        # Split the inputs and targets into training, validation, and testing sets
        self.split(x,y)
   
    def split(self, x, y):

        print(x.shape)
        print(y.shape)

        examples = x.shape[0]
        split1 = int(examples*.6)
        split2 = int(examples*.8)

        self._x_train = x[:split1] 
        self._y_train = y[:split1]

        print("inputs", self._x_train[0])
        print("targets", self._y_train[0])

        self._x_val = x[split1:split2]
        self._y_val = y[split1:split2]

        self._x_test = x[split2:]
        self._y_test = y[split2:]


    def createModel(self):
        # build the model: a single LSTM
        print('Build model...')

        self._model = Sequential()
        self._model.add(Embedding(self._vocabLen, self._embedDepth, input_length=1))
        self._model.add(LSTM(50))
        self._model.add(Dense(self._vocabLen, activation='softmax'))

        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def generateLyrics(self, seed, lyricLen):

        #Initialize the input and output
        if not seed.split()[-1] in self._terms:
            inputText, result = "UNK", " ".join(seed.split()[:-1] + ["UNK"])
        else:
            inputText, result = seed.split()[-1], seed

        # generate a fixed number of words
        for _ in range(lyricLen):
            
            # encode the text as integer
            encoded = np.array(self._tokenizer.texts_to_sequences([inputText])[0])

            # predict a word in the vocabulary
            preds = self._model.predict_classes(encoded, verbose=0)

            # map predicted word index to word
            outputWord = [word for word, index in self._tokenizer.word_index.items() if index==tpreds][0]

            # append to input
            inputText, result = outputWord, result + ' ' + outputWord
	    
        return self.postProcess(result.split())

    def postProcess(self, output):

        with open(self._ENDINGS_CSV, "r") as csvFile:
            r = csv.reader(csvFile, delimiter=',')
            endings = []
            for line in r:
                endings += line
            endings = sorted(list(set(endings)))
            
        newOutput = []
        addEnd = False
        addEndCount = 0
        for i in range(len(output)-1, -1, -1):
            #if token is not an ending and not waiting to append
            if not output[i] in endings and addEnd:
                if output[i][len(output[i])-1] == "t" and output[i+1] == "ce":
                    output[i] = output[i][:len(output[i])-1]
                elif output[i][len(output[i])-1] == "t" and output[i+1] == "ion":
                    output[i] = output[i][:len(output[i])-1] + "s"
                word = ""
                for j in range(addEndCount+1):
                    word += output[i+j]
                newOutput = [word] + newOutput
                addEnd = False
                addEndCount = 0
            elif not output[i] in endings and not addEnd:
                newOutput = [output[i]] + newOutput
            elif output[i] in endings:
                addEnd = True
                addEndCount += 1
        excerpt = newOutput

        final = []
        tagged = nltk.pos_tag(excerpt)
        nnPossibilities = ["this", "that", "the", "a"]
        nnsPossibilities = ["these", "those", "the", "some"]
        for k in range(len(tagged)-1, -1, -1):
            addOrNo = random.randint(0,10)
            if tagged[k][1] == "NN":
                if k == 0:
                    if tagged[k][0] != "i":
                        final = [random.choice(nnPossibilities)] + [tagged[k][0]] + final
                    else:
                        final = [tagged[k][0]] + final
                elif (tagged[k-1][1] != "JJ" or tagged[i-1][1] != "NN") and tagged[k-1][1] != "," and addOrNo < 7:
                    final = [random.choice(nnPossibilities)] + [tagged[k][0]] + final
                else:
                    final = [tagged[k][0]] + final
            elif tagged[i][1] == "NNS":
                if k == 0:
                    final = [random.choice(nnsPossibilities)] + [tagged[k][0]] + final
                elif (tagged[k-1][1] != "JJ" or tagged[k-1][1] != "NN") and tagged[k-1][1] != "," and addOrNo < 7:
                    final = [random.choice(nnsPossibilities)] + [tagged[k][0]] + final
                else:
                    final = [tagged[k][0]] + final
            elif tagged[k][1] == "VBP" or tagged[k][1] == "VBN":
                if tagged[k+1][1] != "NN" and tagged[k+1][1] != "NS" and tagged[k+1][1] != "DT":
                    final = [tagged[k][0]] + ["you"] + final
                else:
                    final = [tagged[k][0]] + final
            else:
                final = [tagged[k][0]] + final
        return " ".join(final)

    def train(self):
        self._model.fit(self._x_train, self._y_train,
          batch_size=self._batchSize,
          validation_data=(self._x_val, self._y_val),
          epochs=self._epochs)

    def evaluate(self):
        score, acc = self._model.evaluate(self._x_test, self._y_test)
        return acc

model = Generator()

def main():
    
    m = model._model
    m.summary()
    print ("Inputs: {}".format(m.input_shape))
    print ("Outputs: {}".format(m.output_shape))

    model.train()
    print("Testing Accuracy", model.evaluate())
    print(model.generateLyrics("filthy", 10))
    print(model.generateLyrics("watermelon", 10))
    print(model.generateLyrics("geronimo", 10))
    print(model.generateLyrics("mambo", 10))
    
if __name__ == "__main__":
    main()
