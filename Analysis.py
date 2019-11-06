import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import io
import scipy.stats
import math
def loadwords():
    words = pd.read_csv("some.csv",header=None)
def countVector(lyrics,word):
    wordCount = np.ones(lyrics.shape)
    for i in range(lyrics.size):
        wordCount[i] = lyrics[i].count(word)
    return wordCount
def linReg(x,y,xlabel = "",ylabel = "",title = ""):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x.astype(float), y.astype(float))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(peak, loveCount, s=.3)
    xmin = np.min(x)
    xmax = np.max(x)
    x2 = range(math.floor(xmin), math.ceil(xmax) + 1)
    y2 = x2 * slope + intercept
    plt.plot(x2, y2)
    plt.show()
    return slope, intercept, r_value, p_value, std_err
data = pd.read_csv("billboard_w_lyrics.csv")
data = data.to_numpy()
print(data.shape)
peak = data[:,3]
weeks = data[:,5]
lyrics = data[:,-1]
loveCount = countVector(lyrics,"I")
#print(linReg(loveCount,peak,"Number of times I appeared in song", "Peak ranking", "Association between the word I and peak ranking"))
print(linReg(loveCount,weeks,"Number of times Love appeared in song", "weeks", "Association between the word I and top weeks"))