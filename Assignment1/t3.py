from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import re

df = pd.read_csv('./a01_spam.csv')
messageList = df['Message'].tolist()

numberOfSpam = 0
numberOfHam = 0
for category, message in df.iterrows():
    if message['Category'] == 'spam':
        numberOfSpam += 1
    else:
        numberOfHam += 1

emailList = {}
emailInSpam = 0
emailInHam = 0
percentEmailInSpam = 0
percentEmailInHam = 0
for category, message in df.iterrows():
    tempList = re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-zA-Z]{2,}\b", message['Message'])
    if len(tempList) > 0:
        if message['Category'] == 'spam':
            emailInSpam += 1
        else:
            emailInHam += 1
    for email in tempList:
        if email in emailList:
            emailList[email] += 1
        else:
            emailList[email] = 1
print("Email:Count")
for email in emailList:
    print(email + ":" + str(emailList[email]))
print("Email in spam: " + str(emailInSpam))
print("Email in ham: " + str(emailInHam))
print("Total Emails: " + str(emailInHam + emailInSpam))
percentEmailInSpam = emailInSpam*100/numberOfSpam
percentEmailInHam = emailInHam*100/numberOfHam
print("Percentage of Email in spam: " + str(percentEmailInSpam) + "%")
print("Percentage of Email in ham: " + str(percentEmailInHam) + "%")

numbersList = {}
numberInSpam = 0
numberInHam = 0
percentNumberInSpam = 0
percentNumberInHam = 0
for category, message in df.iterrows():
    numberList = re.findall(
        r"\b[0]?\d{3}[\s-]?\d{3}[\s-]?\d{4}\b|\b[0]?\d{3}[\s-]?\d{4}[\s-]?\d{3}\b|\b[0]?\d{5}[\s-]?\d{5}\b|\b[0]?\d{4}[\s-]?\d{3}[\s-]?\d{3}\b", message['Message'])
    for number in numberList:
        if number in numbersList:
            numbersList[number] += 1
        else:
            numbersList[number] = 1
    if len(numberList) > 0:
        if message['Category'] == 'spam':
            numberInSpam += 1
        else:
            numberInHam += 1
print("Numbers:Count")
for number in numbersList:
    print(number + ":" + str(numbersList[number]))
print("Numbers in spam: " + str(numberInSpam))
print("Numbers in ham: " + str(numberInHam))
print("Total Number of Numbers: " + str(numberInHam + numberInSpam))
percentNumberInSpam = numberInSpam*100/numberOfSpam
percentNumberInHam = numberInHam*100/numberOfHam
print("Percentage of Numbers in spam: " + str(percentNumberInSpam) + "%")
print("Percentage of Numbers in ham: " + str(percentNumberInHam) + "%")
