from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import re

df = pd.read_csv('./a01_spam.csv')
messageList = df['Message'].tolist()


def isCapital(word):
    if word == "'" or word == "":
        return False
    for i in word:
        if i == "'" and len(word) != 1:
            continue
        if not i.isalpha() or i.islower():
            return False
    return True


spamTotal = 0
spamCap = 0
hamTotal = 0
hamCap = 0
for category, message in df.iterrows():
    if message['Category'] == 'spam':
        words = re.split(
            "\s|(?<!\d)[\\\,\.\/\!\:\?\(\)\[\]\-\&](?!\d)", message['Message'])
        for word in words:
            if len(word) and isCapital(word):
                spamCap += 1
            if len(word):
                spamTotal += 1
    else:
        words = re.split(
            "\s|(?<!\d)[\\\,\.\/\!\:\?\(\)\[\]\-\&](?!\d)", message['Message'])
        for word in words:
            if len(word) and isCapital(word):
                hamCap += 1
            if len(word):
                hamTotal += 1

print("Spam Total: ", spamTotal)
print("Spam Cap: ", spamCap)
print("Total Ham: ", hamTotal)
print("Ham Cap: ", hamCap)

percentCapInSpam = spamCap * 100 / spamTotal
percentCapInHam = hamCap * 100 / hamTotal

print("Percentage of capitalized words in Spam messages = " +
      str(percentCapInSpam) + " %")
print("Percentage of capitalized words in Ham messages = " +
      str(percentCapInHam) + " %")
