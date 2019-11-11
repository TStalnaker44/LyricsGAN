"""
TODO @ Coletta:
- add back in the UNK tokens
- limit vocabulary
- save in json format
- save as pickle
"""
import json
import nltk
from nltk.tokenize import MWETokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import os
import csv
from gensim.models import Word2Vec
import numpy as np

LYRICS_FOLDER = "JSON_Files"
VOCAB_SIZE = 10000
TOKENIZED_CSV = "tokenizedlyrics.csv"
VECTORIZED_CSV = "vectorizedlyrics.csv"
WORD_VECTOR_SIZE = 5
TOKEN = False
EMBED = True
RELOAD = True

class LyricsCleaner:

    def __init__(self, filename):
        self._filename = filename
        self._tokenizer = tokenizer = MWETokenizer([('[', 'Verse', '1', ']'), ('[', 'Verse', '2', ']'), ('[', 'Verse', '3', ']'), ('[', 'Verse', '1', ':',']'), ('[', 'Verse', '2', ':', ']'), ('[', 'Verse', '3', ':', ']'), ('[', 'Pre-Chorus', ']'), ('[', 'Chorus', ']'), ('[', 'Post-Chorus', ']'), ('[', 'Bridge', ']'), ('[', 'Intro', ']')])
        tokenizer.add_mwe(('[', 'Outro', ']'))
        tokenizer.add_mwe(('[', 'Hook', ']'))
        tokenizer.add_mwe(('[', 'Pre-Hook', ']'))

    def tokenizeSong(self):
        with open(self._filename) as songJSON:
            rawData = json.load(songJSON)
            #get the lyrics from the json file
            lyrics = rawData["songs"][0]["lyrics"]
            if not lyrics == None:
                #preserve the newline for prediction
                preserveNewline = lyrics.replace("\n", " NEWLINE ")
                #tokenize the lyrics
                tokenizedLyrics = nltk.word_tokenize(preserveNewline)
                #bring the mwe expressions back together
                tokenizedLyrics = self._tokenizer.tokenize(tokenizedLyrics)
                #add start token
                newLyrics = ['START']
                for word in tokenizedLyrics:
                    #if word is not a stopword, keep it
                    #how is it gonna look loke lyrics if we don't have stop words?
                    if word not in nltk.corpus.stopwords.words("english"):
                        #should we make everything lowercase because capitalization doesn't really matter in songs?
                        newLyrics += [word.lower()]
                #add end token to the end of a song
                newLyrics += ['END']
                return newLyrics

class VocabularyEmbedding:
    def __init__(self, filename):
        self._filename = filename
        #load file
        self._data = None
        with open(filename, "r") as csvFile:
             r = csv.reader(csvFile, delimiter=',')
             self._data = []
             for line in r:
                 self._data.append(line)
             print(len(self._data))
             self._vocab = None
             self.generateVocab()
             self._embedding = None
             self.generateEmbedding()

    def generateVocab(self):
        data = []
        for line in self._data:
            data += line
        fdist = FreqDist(data)
        #only keep the 8000 vocab words with the highest frequencies and the start and end tokens kept
        vocabFreqs = fdist.most_common(VOCAB_SIZE + 2)
        vocab = []
        for freqEntry in vocabFreqs:
            #get the word portion of the frequency entries
            vocabWord = freqEntry[0]
            vocab.append(vocabWord)
        print("Vocab of size ", len(vocab), " generated.")
        self._vocab = vocab
        #print(self._vocab)

    def addUNKTokens(self):
        for j in range(len(self._data)):
            for i in range(len(self._data[j])):
                #replace every word not in the vocab with an unknown token
                if self._data[j][i] not in self._vocab:
                    self._data[j][i] = 'UNK'

    def generateEmbedding(self):
        #default min_count is 5
        self._embedding = Word2Vec(self._data, size=WORD_VECTOR_SIZE, min_count=1)
        words = list(self._embedding.wv.vocab)

    def vectorizeWords(self):
        #replace the words in dataset with vectors to be fed into RNN
        embeddedData = []
        #notRecognized = []
        for j in range(len(self._data)):
            embeddedData.append([])
            for i in range(len(self._data[j])):
                if self._data[j][i] in self._embedding:
                    embeddedData[j].append(self._embedding[self._data[j][i]])
                #else:
                #    embeddedData[j].append(self._data[j][i])
                #    if not self._data[j][i] in notRecognized:
                #        notRecognized.append(self._data[j][i])
        return embeddedData


def writeToCSV(csvFilename, data):
    with open(csvFilename, "a") as csvFile:
         wr = csv.writer(csvFile)
         wr.writerow(data)

def writeNumPytoCSV(csvFilename, data):
    #with open(csvFilename, "a") as csvFile:
         #wr = csv.writer(csvFile)
         #wr.writerow(data)
    numpy.savetxt(csvFilename, data)

def cleanSongsByArtist(currentArtist):
    songs = os.listdir(os.path.join(LYRICS_FOLDER, currentArtist))
    #print(songs)
    for song in songs:
        if song != ".DS_Store":
            print(song)
            parser = LyricsCleaner(os.path.join(LYRICS_FOLDER, currentArtist, song))
            tokenizedLyrics = parser.tokenizeSong()
            if tokenizedLyrics != None:
                writeToCSV(TOKENIZED_CSV, tokenizedLyrics)

def vectorizeLyrics():
    embedding = VocabularyEmbedding(TOKENIZED_CSV)
    vectorizedLyrics = embedding.vectorizeWords()
    #count = 0
    #for song in vectorizedLyrics:
    print(vectorizedLyrics[0])
    #writeToCSV(VECTORIZED_CSV, vectorizedLyrics[0])
    np.savetxt(VECTORIZED_CSV, vectorizedLyrics[0])
    print(vectorizedLyrics[1])
    #writeToCSV(VECTORIZED_CSV, vectorizedLyrics[1])
    np.savetxt(VECTORIZED_CSV, vectorizedLyrics[1])
        #if count > 5:
        #    break
        #count+= 1

def main():
    #create tokenized csv
    if TOKEN:
        artists = os.listdir(os.path.join(LYRICS_FOLDER))
        for artist in artists:
            if artist != ".DS_Store":
                currentArtist = artist #s[0]
                cleanSongsByArtist(currentArtist)
    if EMBED:
        vectorizeLyrics()

    if RELOAD:
        with open(VECTORIZED_CSV, "r") as load:
            r = csv.reader(load, delimiter=",")
            data = []
            for line in r:
                data.append(line)
            print(data)

    #embedding.generateVocab()

    #append results to csv file then load the entire csv file in to determine the vocab
    #then determine the vocab size and make an embedding to replace the words with
    #save the embedding model so it is the same every time
    #then write that to another csv file

if __name__ == "__main__":
   main()
