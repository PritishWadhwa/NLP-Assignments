from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import re

df = pd.read_csv('./a01_spam.csv')
messageList = df['Message'].tolist()

numOfMessages = 0
startWord = input("Enter the Start Word: ")
regexPattern = startWord + r"\s.*"
for i in messageList:
    if re.match(regexPattern, i):
        numOfMessages += 1
        print(i)
print("Number of messages: " + str(numOfMessages))
