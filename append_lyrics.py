import lyricsgenius
import pandas as pd 
import re

genius = lyricsgenius.Genius() #insert API token here
genius.verbose = False
genius.skip_non_songs = False

data = pd.read_csv('billboard.csv')
data_file = open('billboard_w_lyrics.csv', 'a')


#data.insert(len(data.columns), column='Lyrics', value=["null"]*len(data))

print(data.head())

num_songs = 100
current_count = 0

for idx, row in data.iterrows():
    song = genius.search_song(row['Title'], row['Artist'])
    if song is None:
        continue
    row['Lyrics'] = re.sub(r'\[(.*?)\]', '', song.lyrics).replace('\n', ' ').replace('\r', '')
    data_file.write(','.join(map(str, list(row.values))) + '\n')
    current_count += 1
    if current_count >= num_songs:
        break

