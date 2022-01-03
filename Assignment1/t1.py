from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import re

df = pd.read_csv('./a01_spam.csv')
messageList = df['Message'].tolist()

wordsStartingWithVowels = 0
wordsStartingWithConsonents = 0
for i in messageList:
    words = re.split("\s|(?<!\d)[,./!:?\(\)\[\]\-\&](?!\d)", i)
    for word in words:
        if len(word) and word[0].lower() in 'aeiou':
            wordsStartingWithVowels += 1
        elif len(word) and word[0].lower() in 'qwrtypsdfghjklzxcvbnm':
            wordsStartingWithConsonents += 1

print("Words starting with vowels: ", wordsStartingWithVowels)
print("Words starting with consonants: ", wordsStartingWithConsonents)
