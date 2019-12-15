import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split


def load_data_genre():
	spg = 100 #sample per genre
	words = 1000
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
	full_word_freq = np.ones((spg*5, words),dtype = np.int8)
	y = np.ones((spg*5,),dtype = np.int8)
	for i in range(5):
		y[i*spg:(i+1)*spg] = y[i*spg:(i+1)*spg] * i

	full_word_freq[current_index:current_index+spg,:]= hip_pop[:spg,4:4+words]
	current_index += spg
	full_word_freq[current_index:current_index+spg, :] = pop[:spg,4:4+words]
	current_index += spg
	full_word_freq[current_index:current_index+spg, :] = rock[:spg,4:4+words]
	current_index += spg
	full_word_freq[current_index:current_index+spg, :] = country[:spg,4:4+words]
	current_index += spg
	full_word_freq[current_index:current_index+spg, :] = folk[:spg,4:4+words]
	current_index += spg
	return full_word_freq, y


def linearRegression(x,y):
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
	#linRegModel = LinearRegression().fit(X_train,y_train)
	linRegModel = LogisticRegression().fit(X_train, y_train)
	train_score = linRegModel.score(X_train,y_train)
	validation_score = linRegModel.score(X_test, y_test)
	print("test\\ validation score: {} {}".format(train_score, validation_score))


#x_list, y_list = load_data_popularity_listed()
word_freq,y = load_data_genre()
print(word_freq.shape)
print(y.shape)
normalizer = Normalizer()
normalizer.transform(word_freq)
'''
popularity_avg = np.average(popularity_freq_vector)
popularity_std = np.std(popularity_freq_vector)
popularity_freq_vector = (popularity_freq_vector-popularity_avg)/(popularity_std**.5)
'''
word_list = pd.read_csv('filtered_wordlist')
word_list = word_list.to_numpy()
word_list = word_list[:,0]

linearRegression(word_freq,y)