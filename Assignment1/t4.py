from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import re

df = pd.read_csv('./a01_spam.csv')
messageList = df['Message'].tolist()
moneyList = []
spamWithMoney = 0
hamWithMoney = 0
numberOfSpam = 0
numberOfHam = 0
for category, message in df.iterrows():
    if message['Category'] == 'spam':
        numberOfSpam += 1
    else:
        numberOfHam += 1
for category, message in df.iterrows():
    j = message['Message'].replace(",", "")
    tempList = re.findall(
        r"(?:\$|\¥|\€|\¢|\£|\₹|Rs\.|rs\.)\s?\d+(?:\.\d+)?(?:(?:/|\sper\s)(?:min|hr|sec|pax|s|month|wk|msg|day|SMS|m|tone|Msg))?|\d+(?:\.\d+)?p[\/\.](?:min|hr|sec|pax|s|month|wk|msg|day|SMS|m|tone|Msg)?|\d+(?:\.\d+)?\s?[Pp]ounds?|\d+(?:\.\d+)?p[^m]", j)
    for i in tempList:
        moneyList.append(i)
    if len(tempList) > 0:
        if message['Category'] == 'spam':
            spamWithMoney += 1
        else:
            hamWithMoney += 1
    for money in tempList:
        moneyList.append(money)
percentSpamWithMoney = spamWithMoney*100/numberOfSpam
percentHamWithMoney = hamWithMoney*100/numberOfHam
for i in moneyList:
    print(i)
print("Percentage of Spam messages with monetory quantity = " +
      str(percentSpamWithMoney) + "%")
print("Percentage of Ham messages with monetory quantity = " +
      str(percentHamWithMoney) + "%")
