import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer


def load_data_genre():
	freq_vectors = []
	popularity = []
	hip_pop = pd.read_csv('vectorizations/hip-hop_vectorized.csv')
	pop = pd.read_csv('vectorizations/pop_vectorized.csv')
	print(pop.head())
	rock = pd.read_csv('vectorizations/rock_vectorized.csv')
	country = pd.read_csv('vectorizations/vectorized_country.csv')
	folk = pd.read_csv('vectorizations/vectorized_folk.csv')
	hip_pop = hip_pop.to_numpy()
	pop = pop.to_numpy()
	rock = rock.to_numpy()
	country = country.to_numpy()
	folk = folk.to_numpy()
	total_songs = 0
	total_songs += hip_pop.shape[0]
	total_songs += pop.shape[0]
	total_songs += rock.shape[0]
	total_songs += country.shape[0]
	total_songs += folk.shape[0]
	current_index = 0
	full_word_freq = np.ones((total_songs, folk.shape[1] - 4),dtype = np.int8)
	y = np.ones((total_songs,),dtype = np.int8)
	y[500:1000] = y[500,1000]*2
	y[1000:1500] = y[1000, 1500] * 3
	y[1500:2000] = y[1500, 2000] * 4
	y[2000:2500] = y[2000, 2500] * 5

	full_word_freq[current_index:current_index+500,:]= hip_pop[:,4:]
	current_index += hip_pop.shape[0]
	full_word_freq[current_index:current_index+500, :] = pop[:, 4:]
	current_index += pop.shape[0]
	full_word_freq[current_index:current_index+500, :] = rock[:, 4:]
	current_index += rock.shape[0]
	full_word_freq[current_index:current_index+500, :] = country[:, 4:]
	current_index += country.shape[0]
	full_word_freq[current_index:current_index+500, :] = folk[:, 4:]
	current_index += folk.shape[0]
	return full_word_freq,y





#x_list, y_list = load_data_popularity_listed()
word_freq,y = load_data_genre()
normalizer = Normalizer()
#normalizer.transform(word_freq)
'''
popularity_avg = np.average(popularity_freq_vector)
popularity_std = np.std(popularity_freq_vector)
popularity_freq_vector = (popularity_freq_vector-popularity_avg)/(popularity_std**.5)
'''
word_list = pd.read_csv('filtered_wordlist')
word_list = word_list.to_numpy()
word_list = word_list[:,0]