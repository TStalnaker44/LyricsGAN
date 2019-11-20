"""
Data cleaning by Abby, Coletta, and Angela
TO TURN IN:
- this file
- code to pull genius lyrics
- ten of raw data files
- tokenizedlyrics.csv for ten songs
- name csv
- output file of ten songs
- write up
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
EMBED = True
RELOAD = False
SIGNAL_WORDS = ['Verse', 'Pre-Chorus', 'Chorus', 'Post-Chorus', 'Bridge', 'Intro', 'Outro', 'Hook', 'Pre-Hook']
#COMP_CHARS = string.printable + "“…’”’’‘" + "—"

class LyricsCleaner:
    """cleans and tokenizes the from a song lyrics in preparation for embedding"""

    def __init__(self, filename):
        """initializes a LyricsCleaner object"""
        self._filename = filename
        self._tokenizer = MWETokenizer()
        for word in SIGNAL_WORDS:
            self._tokenizer.add_mwe(('[', word, ']'))
        self._stemmer = LancasterStemmer()

    def tokenizeSong(self):
        """breaks up the lyrics into tokens using the nltk tokenizer, stemming,
        and various normalization techniques"""
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
                #preserve the newline for prediction, want to predict the newline character
                preserveNewline = lyrics.replace("\n", " **NEWLINE** ")
                #tokenize the lyrics
                tokenizedLyrics = nltk.word_tokenize(preserveNewline)
                #replace people's names with general name token
                for k in range(len(tokenizedLyrics)):
                    if tokenizedLyrics[k] in names:
                        tokenizedLyrics[k] = "**NAME_VAR**"
                    #NOT DOING THIS ANYMORE: take out words that are not english
                    #else:
                    #    for h in range(len(tokenizedLyrics[k])):
                    #        if not tokenizedLyrics[k][h] in string.printable and len(tokenizedLyrics[k]) > 1 and tokenizedLyrics[k][h] != "…":
                    #            if h != len(tokenizedLyrics[k]):
                    #                print(tokenizedLyrics[k] + " ==> " + tokenizedLyrics[k][h])
                    #                tokenizedLyrics[k] = "**NOT_ENLGISH**"
                #bring the multi-word expressions back together ([CHOURS], [VERSE], etc)
                tokenizedLyrics = self._tokenizer.tokenize(tokenizedLyrics)
                #add start token
                newLyrics = ['START']

                #normalize the labels for the parts of the song
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
                    elif word not in nltk.corpus.stopwords.words("english"):
                        if not word[2:len(word)-2] == SIGNAL_WORDS:
                            #make everything lowercase because capitalization doesn't really matter in songs?
                            #add the stem
                            newLyrics += [self._stemmer.stem(word.lower())]
                            if word.lower() != self._stemmer.stem(word.lower()):
                                #if stem is same as original word except for last letter in original
                                if word.lower()[:len(word)-1] == self._stemmer.stem(word.lower()):
                                    #add the last letter in the original word
                                    newLyrics += word.lower()[len(word)-1:]
                                #if stem is same as original word except for last two letters in original
                                elif word.lower()[:len(word)-2] == self._stemmer.stem(word.lower()):
                                    #add the last two letters in the original word
                                    newLyrics += word.lower()[len(word)-2:]
                                #if stem is same as original word except for last three letters in original
                                elif word.lower()[:len(word)-3] == self._stemmer.stem(word.lower()):
                                    #add the last three letters in the original word
                                    newLyrics += word.lower()[len(word)-3:]
                                #if stem is like once or since the stem is "ont" or "sint"
                                elif word.lower()[len(word)-3:len(word)-1] == "ce" and self._stemmer.stem(word.lower())[len(self._stemmer.stem(word.lower()))-1] == "t":
                                    #add the "ce" as a token
                                    newLyrics += word.lower()[len(word)-3:len(word)-1]
                    i += 1
                                    #print(word.lower()[len(word)-3:len(word)-1])
                #add end token to the end of a song
                newLyrics += ['END']
                return newLyrics

class VocabularyEmbedding:
    """creates an embedding for the dataset"""
    def __init__(self, filename):
        """initializes a VocabularyEmbedding object"""
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
        """generates the vocabulary for the dataset"""
        #create a vocabulary of only the words that appear more than once in the entire dataset
        embedding = Word2Vec(self._data, size=WORD_VECTOR_SIZE, min_count=2)
        self._vocab = list(embedding.wv.vocab)
        print(len(self._vocab))

    def addUNKTokens(self):
        """replaces words not in the vocabulary with UNK tokens"""
        for j in range(len(self._data)):
            for i in range(len(self._data[j])):
                #replace every word not in the vocab with an unknown token
                if self._data[j][i] not in self._vocab:
                    #print(self._data[j][i], "==> UNK")
                    self._data[j][i] = 'UNK'

    def generateEmbedding(self):
        """generates the embedding for the dataset given the data"""
        #add the unknown tokens into the dataset
        self.addUNKTokens()
        #create an embedding of only the words that are mentioned more than once in the dataset, start, end, and unk tokens
        self._embedding = Word2Vec(self._data, size=WORD_VECTOR_SIZE, min_count=1)
        self._embedding.wv.save_word2vec_format('wordvectors.bin')

    def vectorizeWords(self):
        """replaces the words in the preprocessed, tokenized dataset with their word vectors"""
        #replace the words in preprocessed dataset with vectors to be fed into RNN
        embeddedData = []
        for j in range(len(self._data)):
            embeddedData.append([])
            for i in range(len(self._data[j])):
                if self._data[j][i] in self._embedding:
                    #replace word at a certain position with its vector representation
                    embeddedData[j].append(self._embedding[self._data[j][i]])
        return embeddedData

    def loadEmbedding(self):
        self._embedding = Word2Vec.load('wordvectors.bin')


def writeToCSV(csvFilename, data):
    """writes data to a csv file given the file name"""
    with open(csvFilename, "a") as csvFile:
         wr = csv.writer(csvFile)
         wr.writerow(data)

def writeVectortoJSON(jsonFilename, data):
    """writes the vectorized lyrics to a json file given the file name"""
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
                        #print(song)
                        components = song.split("_")
                        songName = components[2][:len(componenets[2])-5]
                        output[currentArtist][songName] = data[0]

                        parser = LyricsCleaner(os.path.join(LYRICS_FOLDER, currentArtist, song))
                        tokenizedLyrics = parser.tokenizeSong()
                        if tokenizedLyrics != None:
                            writeToCSV(TOKENIZED_CSV, tokenizedLyrics)
                            i+=1
    numpy.savetxt(csvFilename, data)

class FinalOutput:
    """saves the final tokenized, vectorized output"""
    def __init__(self, filename):
        """initializes a FinalOutput object"""
        self._filename = filename
        self._output = {}

    def getData(self):
        """returns the output"""
        return self._output

    def addArtist(self,artist):
        """creates a new entry in the output given the artist"""
        self._output[artist] = {}

    def addSongByArtist(self, artist, song):
        """saves a new song by an artist given the artist"""
        components = song.split("_")
        songName = components[2][:len(components[2])-5]
        self._output[artist][songName] = {"lyrics": []}

    def addLyricsByArtistAndSong(self, artist, song, lyrics):
        """saves the lyrics of a song given the artist and the song"""
        #lyrics should be a numpy array
        reshapedLyrics = np.reshape(lyrics, (len(lyrics), WORD_VECTOR_SIZE, 1))
        #print(reshapedLyrics)
        #print(self._output)
        self._output[artist][song]["lyrics"] = reshapedLyrics.tolist()

    def writeFormatJSON(self):
        """writes the format of the output to a json file"""
        with open("outputformat.json", 'w') as outputFile:
            json.dump(self._output, outputFile)

    def getOutputFormat(self):
        """returns output format from format json file"""
        with open("outputformat.json", 'r') as inputFile:
            self._output = json.load(inputFile)
        return self._output

    def writeToJSON(self):
        """writes the final output to be fed back into an RNN to a json file"""
        with open(self._filename, 'w') as outputFile:
            json.dump(self._output, outputFile)


def cleanSongsByArtist(currentArtist, outputJson):
    """clean each song by an artist"""
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
    """vectorize the songs in the dataset given the output format"""
    format = outputJson.getOutputFormat()
    #print(format)
    embedding = VocabularyEmbedding(TOKENIZED_CSV)
    print("Starting lyric vectorization.")
    vectorizedLyrics = embedding.vectorizeWords()
    print("Done with lyric vectorization.")
    i=0
    #run this on a lab machine because it will take a long af time otherwise
    for artist in format:
        print(artist)
        for song in format[artist]:
            print("\t" + song)
            lyrics = vectorizedLyrics[i]
            outputJson.addLyricsByArtistAndSong(artist, song, lyrics)
            i+=1
    outputJson.writeToJSON()

def exampleOutput(outputJson):
    exampleArtists = ["ariana grande", "taylor swift", "katy perry", "selena gomez", "britney spears"]
    exampleSongs = ["7rings", "newromantics", "cruelsummer", "teenagedream", "lookathernow", "circus"]
    format = outputJson.getOutputFormat()
    #print(format)
    embedding = VocabularyEmbedding(TOKENIZED_CSV)
    print("Starting lyric vectorization.")
    vectorizedLyrics = embedding.vectorizeWords()
    print("Done with lyric vectorization.")
    i=0
    #right now just does it for the first justin timberlake song because of the break statements
    for artist in format:
        if artist in exampleArtists:
            #print(artist)
            for song in format[artist]:
                if song in exampleSongs:
                    #print("\t" + song)
                    #print(i)
                    lyrics = vectorizedLyrics[i]
                    outputJson.addLyricsByArtistAndSong(artist, song, lyrics)
                i+=1
    with open(VECTORIZED_JSON, 'w') as outputFile:
        fullData = outputJson.getData()
        exData = {}
        for artist in exampleArtists:
            if not artist in exData:
                exData[artist] = {}
            for song in exampleSongs:
                if song in fullData[artist]:
                    exData[artist][song] = fullData[artist][song]
        json.dump(exData, outputFile)

def main():
    """change the constants TOKEN, EMBED, and RELOAD to true if you want to run that section"""
    outputJson = FinalOutput(VECTORIZED_JSON)
    #create csv of tokenized words
    if TOKEN:
        artists = os.listdir(os.path.join(LYRICS_FOLDER))
        for artist in artists:
            if artist != ".DS_Store":
                currentArtist = artist
                outputJson.addArtist(currentArtist)
                cleanSongsByArtist(currentArtist, outputJson)
    #create the embedding for the dataset
    if EMBED:
        vectorizeLyrics(outputJson)
        #exampleOutput(outputJson)
    #test that the data can be easily loaded back in to be processed
    if RELOAD:
        with open(VECTORIZED_JSON, "r") as load:
            preprocessedData = json.load(load)
            data = []
            for singer in preprocessedData:
                for song in preprocessedData[singer]:
                    loadedLyrics = np.array(preprocessedData[singer][song]["lyrics"])
                    print(loadedLyrics.shape)
                    data.append(loadedLyrics)
            #print(data)

if __name__ == "__main__":
   main()
