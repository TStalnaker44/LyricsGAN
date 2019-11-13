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
from nltk.stem.lancaster import LancasterStemmer
import nltk.classify.textcat
import string

LYRICS_FOLDER = "JSON_Files"
VOCAB_SIZE = 10000
TOKENIZED_CSV = "tokenizedlyrics.csv"
VECTORIZED_JSON = "vectorizedlyrics.json"
WORD_VECTOR_SIZE = 5
NAMES_CSV = "names.csv"
TOKEN = True
EMBED = False
RELOAD = False
SIGNAL_WORDS = ['Verse', 'Pre-Chorus', 'Chorus', 'Post-Chorus', 'Bridge', 'Intro', 'Outro', 'Hook', 'Pre-Hook']
#COMP_CHARS = string.printable + "“…’”’’‘" + "—"

class LyricsCleaner:

    def __init__(self, filename):
        self._filename = filename
        #self._tokenizer = tokenizer = MWETokenizer([('[', 'Verse', '1', ']'), ('[', 'Verse', '2', ']'), ('[', 'Verse', '3', ']'), ('[', 'Verse', '1', ':',']'), ('[', 'Verse', '2', ':', ']'), ('[', 'Verse', '3', ':', ']'), ('[', 'Pre-Chorus', ']'), ('[', 'Chorus', ']'), ('[', 'Post-Chorus', ']'), ('[', 'Bridge', ']'), ('[', 'Intro', ']')])
        #self._tokenizer = tokenizer = MWETokenizer([('[', 'Verse'), ('[', 'Pre-Chorus'), ('[', 'Chorus'), ('[', 'Post-Chorus'), ('[', 'Bridge'), ('[', 'Intro')])
        self._tokenizer = MWETokenizer()
        for word in SIGNAL_WORDS:
            self._tokenizer.add_mwe(('[', word, ']'))
        #tokenizer.add_mwe(('[', 'Outro'))
        #tokenizer.add_mwe(('[', 'Hook'))
        #tokenizer.add_mwe(('[', 'Pre-Hook'))
        self._stemmer = LancasterStemmer()

    def tokenizeSong(self):
        with open(NAMES_CSV) as nameFile:
            read = csv.reader(nameFile, delimiter=",")
            names = []
            for name in read:
                names.append(name[0])
        with open(self._filename) as songJSON:
            rawData = json.load(songJSON)
            #get the lyrics from the json file
            lyrics = rawData["songs"][0]["lyrics"]
            if not lyrics == None:
                #preserve the newline for prediction
                preserveNewline = lyrics.replace("\n", " **NEWLINE** ")
                #tokenize the lyrics
                tokenizedLyrics = nltk.word_tokenize(preserveNewline)
                for k in range(len(tokenizedLyrics)):
                    if tokenizedLyrics[k] in names:
                        tokenizedLyrics[k] = "**NAME_VAR**"
                    #else:
                    #    for h in range(len(tokenizedLyrics[k])):
                    #        if not tokenizedLyrics[k][h] in string.printable and len(tokenizedLyrics[k]) > 1 and tokenizedLyrics[k][h] != "…":
                    #            if h != len(tokenizedLyrics[k]):
                    #                print(tokenizedLyrics[k] + " ==> " + tokenizedLyrics[k][h])
                    #                tokenizedLyrics[k] = "**NOT_ENLGISH**"
                #bring the mwe expressions back together
                tokenizedLyrics = self._tokenizer.tokenize(tokenizedLyrics)
                #add start token
                newLyrics = ['START']
                test = False
                i = 0
                while i < len(tokenizedLyrics):
                    word = tokenizedLyrics[i]
                    if word == "[":
                        if tokenizedLyrics[i + 1] in SIGNAL_WORDS:
                            j = i + 2
                            while tokenizedLyrics[j] != "]" and j < len(tokenizedLyrics) - 1:
                                j += 1
                            word = word + "_" + tokenizedLyrics[i+1] + "_" + tokenizedLyrics[j]
                            newLyrics += [word.lower()]
                            i = j

                    #if word is not a stopword, keep it
                    #how is it gonna look loke lyrics if we don't have stop words?
                    elif word not in nltk.corpus.stopwords.words("english"):
                        if not word[2:len(word)-2] == SIGNAL_WORDS:
                            #should we make everything lowercase because capitalization doesn't really matter in songs?
                            newLyrics += [self._stemmer.stem(word.lower())]#[word.lower()]
                            if word.lower() != self._stemmer.stem(word.lower()):
                                #print(word.lower(), ": ", self._stemmer.stem(word.lower()))
                                if word.lower()[:len(word)-1] == self._stemmer.stem(word.lower()):
                                    newLyrics += word.lower()[len(word)-1:]
                                    #print(word.lower()[len(word)-1:])
                                elif word.lower()[:len(word)-2] == self._stemmer.stem(word.lower()):
                                    newLyrics += word.lower()[len(word)-2:]
                                    #print(word.lower()[len(word)-2:])
                                elif word.lower()[:len(word)-3] == self._stemmer.stem(word.lower()):
                                    newLyrics += word.lower()[len(word)-3:]
                                    #print(word.lower()[len(word)-3:])
                                elif word.lower()[len(word)-3:len(word)-1] == "ce" and self._stemmer.stem(word.lower())[len(self._stemmer.stem(word.lower()))-1] == "t":
                                    newLyrics += word.lower()[len(word)-3:len(word)-1]
                    i += 1
                                    #print(word.lower()[len(word)-3:len(word)-1])
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
             #print(len(self._data))
             self._vocab = None
             self.generateVocab()
             self._embedding = None
             self.generateEmbedding()

    def generateVocab(self):
        embedding = Word2Vec(self._data, size=WORD_VECTOR_SIZE, min_count=2)
        self._vocab = list(embedding.wv.vocab)
        print(len(self._vocab))
        #data = []
        #for line in self._data:
            #data += line
        #fdist = FreqDist(data)
        #print(len(fdist))
        #only keep the 8000 vocab words with the highest frequencies and the start and end tokens kept
        #vocabFreqs = fdist.most_common(VOCAB_SIZE + 2)
        #vocab = []
        #for freqEntry in vocabFreqs:
            #get the word portion of the frequency entries
            #vocabWord = freqEntry[0]
            #vocab.append(vocabWord)
        #print("Vocab of size ", len(vocab), " generated.")
        #self._vocab = vocab
        #self.addUNKTokens()
        #print(self._data[1])
        #print(self._vocab)

    def addUNKTokens(self):
        for j in range(len(self._data)):
            for i in range(len(self._data[j])):
                #replace every word not in the vocab with an unknown token
                if self._data[j][i] not in self._vocab:
                    print(self._data[j][i], "==> UNK")
                    self._data[j][i] = 'UNK'

    def generateEmbedding(self):
        #default min_count is 5
        self.addUNKTokens()
        self._embedding = Word2Vec(self._data, size=WORD_VECTOR_SIZE, min_count=1)
        #print(len(words))

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

def writeVectortoJSON(jsonFilename, data):
    output = {}
    with open(csvFilename, "a") as csvFile:
        i = 0
        artists = os.listdir(os.path.join(LYRICS_FOLDER))
        for artist in artists:
            if artist != ".DS_Store":
                currentArtist = artist
                output[currentArtist] = {}
                songs = os.listdir(os.path.join(LYRICS_FOLDER, currentArtist))
                #print(songs)
                for song in songs:
                    if song != ".DS_Store":
                        print(song)
                        components = song.split("_")
                        songName = components[2][:len(componenets[2])-5]
                        output[currentArtist][songName] = data[0]

                        parser = LyricsCleaner(os.path.join(LYRICS_FOLDER, currentArtist, song))
                        tokenizedLyrics = parser.tokenizeSong()
                        if tokenizedLyrics != None:
                            writeToCSV(TOKENIZED_CSV, tokenizedLyrics)
                            i+=1
    #with open(csvFilename, "a") as csvFile:
         #wr = csv.writer(csvFile)
         #wr.writerow(data)
    numpy.savetxt(csvFilename, data)

class FinalOutput:
    def __init__(self, filename):
        self._filename = filename
        self._output = {}

    def getData(self):
        return self._output

    def addArtist(self,artist):
        self._output[artist] = {}

    def addSongByArtist(self, artist, song):
        components = song.split("_")
        songName = components[2][:len(components[2])-5]
        self._output[artist][songName] = {"lyrics": []}

    def addLyricsByArtistAndSong(self, artist, song, lyrics):
        #lyrics should be a numpy array
        reshapedLyrics = np.reshape(lyrics, (len(lyrics), WORD_VECTOR_SIZE, 1))
        print(reshapedLyrics)
        print(self._output)
        self._output[artist][song]["lyrics"] = reshapedLyrics.tolist()

    def writeFormatJSON(self):
        with open("outputformat.json", 'w') as outputFile:
            json.dump(self._output, outputFile)

    def getOutputFormat(self):
        with open("outputformat.json", 'r') as inputFile:
            self._output = json.load(inputFile)
        return self._output

    def writeToJSON(self):
        with open(self._filename, 'w') as outputFile:
            json.dump(self._output, outputFile)


def cleanSongsByArtist(currentArtist, outputJson):
    songs = os.listdir(os.path.join(LYRICS_FOLDER, currentArtist))
    #print(songs)
    for song in songs:
        if song != ".DS_Store":
            print(song)
            outputJson.addSongByArtist(currentArtist, song)
            parser = LyricsCleaner(os.path.join(LYRICS_FOLDER, currentArtist, song))
            tokenizedLyrics = parser.tokenizeSong()
            if tokenizedLyrics != None:
                writeToCSV(TOKENIZED_CSV, tokenizedLyrics)
    outputJson.writeFormatJSON()

def vectorizeLyrics(outputJson):
    format = outputJson.getOutputFormat()
    print(format)
    embedding = VocabularyEmbedding(TOKENIZED_CSV)
    print("Starting lyric vectorization.")
    vectorizedLyrics = embedding.vectorizeWords()
    print("Done with lyric vectorization.")
    i=0
    for artist in format:
        print(artist)
        for song in format[artist]:
            print("\t" + song)
            lyrics = vectorizedLyrics[i]
            outputJson.addLyricsByArtistAndSong(artist, song, lyrics)
            i+=1
            break
        break
    outputJson.writeToJSON()
    #count = 0
    #for song in vectorizedLyrics:
    #print(vectorizedLyrics[0])
    #writeToCSV(VECTORIZED_CSV, vectorizedLyrics[0])
    #np.savetxt(VECTORIZED_CSV, vectorizedLyrics[0])
    #print(vectorizedLyrics[1])
    #writeToCSV(VECTORIZED_CSV, vectorizedLyrics[1])
    #np.savetxt(VECTORIZED_CSV, vectorizedLyrics[1])
        #if count > 5:
        #    break
        #count+= 1

def main():
    outputJson = FinalOutput(VECTORIZED_JSON)
    #create tokenized csv
    if TOKEN:
        artists = os.listdir(os.path.join(LYRICS_FOLDER))
        for artist in artists:
            if artist != ".DS_Store":
                currentArtist = artist
                outputJson.addArtist(currentArtist)
                cleanSongsByArtist(currentArtist, outputJson)
    if EMBED:
        vectorizeLyrics(outputJson)

    if RELOAD:
        with open(VECTORIZED_JSON, "r") as load:
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
