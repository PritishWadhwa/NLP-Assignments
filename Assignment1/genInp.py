from nltk.tokenize import word_tokenize, sent_tokenize
inpword = input()
fileName = input()
data = None
with open(fileName, 'r') as f:
    data = f.read()

countWord = 0
countSent = 0

words = word_tokenize(data)
for word in words:
    if word == inpword:
        countWord += 1

sentences = sent_tokenize(data)
for sentence in sentences:
    sentWords = word_tokenize(sentence)
    for sentWord in sentWords:
        if sentWord == inpword:
            countSent += 1

print("Count of words: " + str(countWord))
print("Count of sentences: " + str(countSent))
