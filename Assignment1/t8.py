from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import re

df = pd.read_csv('./a01_spam.csv')
messageList = df['Message'].tolist()
numOfSent = 0
endWord = input("Enter the End Word: ")
regexPattern = r".*\b" + endWord + r"[\.{1,3}\?\!\s\'\"\)\]]?$"
for i in messageList:
    sentences = sent_tokenize(i)
    for sent in sentences:
        if re.match(regexPattern, sent):
            numOfSent += 1
            print(sent)
print("Number of sentences: " + str(numOfSent))
