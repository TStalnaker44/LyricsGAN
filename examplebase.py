
import numpy as np

class ModelData():

    def __init__(self):
        self._inputs = None
        self._targets = None
        self._vocab = []
        self._lyrics = []

        self._hasData = False

    def getInputs(self):
        return self._inputs

    def getTargets(self):
        return self._targets

    def getVocab(self):
        return self._vocab

    def getLyrics(self):
        return self._lyrics

    def setInputs(self, inputs):
        self._inputs = inputs

    def setTargets(self, targets):
        self._targets = targets

    def setVocab(self, vocab):
        self._vocab = vocab

    def setLyrics(self, lyrics):
        self._lyrics = lyrics

    def hasData(self):
        return self._hasData

    def getWords2Indices(self):
        return dict((c, i) for i, c in enumerate(self._vocab))

    def getIndices2Words(self):
        return dict((i, c) for i, c in enumerate(self._vocab))

    def setLoaded(self):
        self._hasData = True

class ExampleBase():

    def __init__(self):

        self._songExamples = {}
        self._songTargets = {}

        self._vocab = []
        self._lyrics = []

        self._allInputs = None
        self._allTargets = None

    def getInput(self, songNumber):
        return self._songExamples[songNumber]

    def getTarget(self, songNumber):
        return self._songTargets[songNumber]

    def getAllInputs(self):
        return self._allInputs

    def getAllTargets(self):
        return self._allTargets

    def setAllTargets(self, targets):
        self._allTargets = targets

    def setAllInputs(self, inputs):
        self._allInputs = inputs

    def songStored(self, songNumber):
        return songNumber in self._songExamples.keys()

    def addInput(self, songNumber, example):
        self._songExamples[songNumber] = example

    def addTarget(self, songNumber, target):
        self._songTargets[songNumber] = target

    def getVocab(self):
        return self._vocab

    def addVocab(self, vocab):
        self._vocab = vocab

    def getLyrics(self):
        return self._lyrics

    def addLyrics(self, lyrics):
        self._lyrics = lyrics

    def hasVocab(self):
        return not len(self._vocab) == 0

class DataBase():

    def __init__(self):
        self._data = None

    def hasData(self):
        return type(self._data) == np.ndarray

    def getData(self):
        return self._data

    def setData(self, data):
        self._data = data
