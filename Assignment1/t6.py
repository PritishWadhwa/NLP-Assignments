from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer

df = pd.read_csv('./a01_spam.csv')
messageList = df['Message'].tolist()

rt = RegexpTokenizer(r"\s|(?<!\d)[,./!:?\(\)\[\]-](?!\d)")
clitics = []
cliticSet = set()
numClitics = 0
for i in messageList:
    words = re.split("\s|(?<!\d)[,./!:?&\(\)\[\]-](?!\d)", i)
    tempList = []
    for word in words:
        tempWord = re.search(r"[a-zA-Z]+\'[a-zA-Z]+", word)
        if tempWord is not None:
            cliticSet.add(tempWord.string[tempWord.start():tempWord.end()])
            clitics.append(tempWord.string[tempWord.start():tempWord.end()])
            numClitics += 1
            tempList.append(tempWord.string[tempWord.start():tempWord.end()])
for clitic in clitics:
    print(clitic)
