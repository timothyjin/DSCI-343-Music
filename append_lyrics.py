import lyricsgenius
import pandas as pd 

genius = lyricsgenius.Genius() #insert token here
genius.verbose = False
genius.skip_non_songs = False

data = pd.read_csv('billboard.csv')

data.insert(len(data.columns), column='Lyrics', value=["null"]*len(data))

print(data.head())

num_songs = 100
current_count = 0

for idx, row in data.iterrows():
    song = genius.search_song(row['Title'], row['Artist'])
    row['Lyrics'] = song.lyrics
    current_count += 1
    if current_count >= num_songs:
        break

data.to_csv()


