import os
import csv
import lyricsgenius
import pandas as pd
import re
import string


genius = lyricsgenius.Genius('') #insert API token here
genius.verbose = False
genius.remove_section_headers = True
genius.skip_non_songs = True

input_file_name = 'samples/pop_sample_counts.csv'
output_file_name = 'samples/pop_sample_lyrics.csv'
if not os.path.exists(input_file_name):
    print('File not found')
    sys.exit(1)

input_data = pd.read_csv(input_file_name, delimiter=',', header=0, quotechar='"')
output_data_file = open(output_file_name, 'a+', encoding='utf-8')
writer = csv.writer(output_data_file, lineterminator='\n')

start_index = int(input("From which song index to start scraping? "))
stop_index = int(input("At which song index to stop scraping? "))

print(input_data.head())

for idx, row in enumerate(input_data.itertuples()):
    if idx < start_index:
        continue
    elif idx > stop_index:
        break
    title = getattr(row, 'Title')
    artist = getattr(row, 'Artist')
    count = getattr(row, 'Count')
    print(idx, ":", artist, "-", title)
    song = genius.search_song(title, artist)
    if song is None:
        continue
    formatted_lyrics = re.sub(r'\[(.*?)\]', ' ', song.lyrics).replace('\n', ' ').replace('\r', ' ').replace('"', "'")
    printable_lyrics = ''.join(filter(lambda x: x in string.printable, formatted_lyrics))
    writer.writerow([title, artist, str(count), printable_lyrics])
