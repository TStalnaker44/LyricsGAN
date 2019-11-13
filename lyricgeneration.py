"""
Abby Nason, Angela Delos Reyes, Coletta Fuller
Collin Glatz,  Max Masaitis, Trevor Stalnaker
lyricgeneration.py

Final Project
"""
import lyricsgenius
import csv
import pandas as pd
import os

genius = lyricsgenius.Genius("fAnxCbFjzcXG6m3puCzR6cA1YtSlhAFNPRf0qIU2fvRCQi2Z9H5JZCC-OIbN4LSX")
artist = genius.search_artist("Alicia Keys")
#song = genius.search_song("Imagine", artist.name) # gets the lyrics
#print(song.lyrics) # prints the song lyrics
os.getcwd()

artist.save_lyrics() # save lyrics as a json file

##def main():
##
##if __name__ == "__main__":
##    main()
