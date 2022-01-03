from tqdm import tqdm
import json
import numpy as np
import os

trainData = './train.txt'
with open(trainData) as f:
    corpus = f.read()
sentences = corpus.splitlines()

validationDataPath = './validation.jsonl'
with open('validation.jsonl') as f:
    validationData = [json.loads(line) for line in f]

wordFreq = {}
totalWords = 0
noBigrams = 0
bigramFreq = {}
removeSet = [',', '.', '?', '!']
for i in tqdm(range(len(sentences))):
    # Preprocessing each sentence
    words = sentences[i].split()
    words = ['<start>'] + words + ['<end>']
    words = [word for word in words if word not in removeSet]
    # Creating unigram frequency map
    for word in words:
        totalWords += 1
        if word not in wordFreq:
            wordFreq[word] = 1
        else:
            wordFreq[word] += 1
    # Creating bigram frequency map
    for j in range(len(words) - 1):
        noBigrams += 1
        bigram = words[j] + ' ' + words[j + 1]
        if bigram not in bigramFreq:
            bigramFreq[bigram] = 1
        else:
            bigramFreq[bigram] += 1

wordSet = set(wordFreq.keys())
bigramSet = set(bigramFreq.keys())


correctPred = 0
totalPred = 0
outFile = './output.txt'
with open(outFile, 'a') as f:
    f.write("Bigram Language Model - Without Smoothing" + os.linesep)
    for i in tqdm(range(len(validationData))):
        lineToAdd = "Question: "
        lineToAdd += validationData[i]['question']
        # Preprocessing each sentence in validation data
        words = validationData[i]['question'].split()
        words = ['<start>'] + words + ['<end>']
        words = [word for word in words if word not in removeSet]
        index = words.index('XXXXX')
        options = validationData[i]['options']
        counts = []
        # Calculating the probability of each option
        for option in options:
            key = str(words[index-1] + ' ' + option)
            if key not in bigramFreq:
                counts.append(0)
            else:
                counts.append(bigramFreq[key]/float(wordFreq[words[index-1]]))
        counts = np.array(counts)
        # Choosing the best option
        pred = np.argmax(counts)
        totalPred += 1
        lineToAdd += " Prediction: "
        lineToAdd += options[pred]
        if options[pred] == validationData[i]['answer']:
            correctPred += 1
        f.write(lineToAdd + os.linesep)
print(f"Accuracy on validataion set: {correctPred * 100 / totalPred}%")

correctPred = 0
totalPred = 0
outFile = './output.txt'
with open(outFile, 'a') as f:
    f.write(os.linesep + "Bigram Language Model - Add-1 Smoothing" + os.linesep)
    for i in tqdm(range(len(validationData))):
        lineToAdd = "Question: "
        lineToAdd += validationData[i]['question']
        words = validationData[i]['question'].split()
        words = ['<start>'] + words + ['<end>']
        words = [word for word in words if word not in removeSet]
        index = words.index('XXXXX')
        options = validationData[i]['options']
        counts = []
        for option in options:
            key = str(words[index-1] + ' ' + option)
            prevWordCount = 0
            if words[index-1] in wordFreq:
                prevWordCount = wordFreq[words[index-1]]
            if key not in bigramFreq:
                counts.append(1/(float(prevWordCount) + len(wordSet)))
            else:
                counts.append((bigramFreq[key] + 1) /
                              float(prevWordCount + len(wordSet)))
        counts = np.array(counts)
        pred = np.argmax(counts)
        totalPred += 1
        lineToAdd += " Prediction: "
        lineToAdd += options[pred]
        if options[pred] == validationData[i]['answer']:
            correctPred += 1
        f.write(lineToAdd + os.linesep)
    print(f"Accuracy on validataion set: {correctPred * 100 / totalPred}%")

K = 0.00001
correctPred = 0
totalPred = 0
outFile = './output.txt'
with open(outFile, 'a') as f:
    f.write(os.linesep + "Bigram Language Model - Add-k Smoothing without using the formula m = k*V" + os.linesep)
    for i in tqdm(range(len(validationData))):
        lineToAdd = "Question: "
        lineToAdd += validationData[i]['question']
        words = validationData[i]['question'].split()
        words = ['<start>'] + words + ['<end>']
        words = [word for word in words if word not in removeSet]
        index = words.index('XXXXX')
        options = validationData[i]['options']
        counts = []
        for option in options:
            key = str(words[index-1] + ' ' + option)
            prevWordCount = 0
            if words[index-1] in wordFreq:
                prevWordCount = wordFreq[words[index-1]]
            if key not in bigramFreq:
                counts.append(K/(float(prevWordCount) + K*len(wordSet)))
            else:
                counts.append((bigramFreq[key] + K) /
                              float(prevWordCount + K*len(wordSet)))
        counts = np.array(counts)
        pred = np.argmax(counts)
        totalPred += 1
        lineToAdd += " Prediction: "
        lineToAdd += options[pred]
        if options[pred] == validationData[i]['answer']:
            correctPred += 1
        f.write(lineToAdd + os.linesep)
    print(f"Accuracy on validataion set: {correctPred * 100 / totalPred}%")

K = 0.00001
correctPred = 0
totalPred = 0
outFile = './output.txt'
with open(outFile, 'a') as f:
    f.write(os.linesep + "Bigram Language Model - Add-k Smoothing With using the formula m = k*V, assuming V is the number of distinct words in vocabulary" + os.linesep)
    for i in tqdm(range(len(validationData))):
        lineToAdd = "Question: "
        lineToAdd += validationData[i]['question']
        words = validationData[i]['question'].split()
        words = ['<start>'] + words + ['<end>']
        words = [word for word in words if word not in removeSet]
        index = words.index('XXXXX')
        words[index-1]
        options = validationData[i]['options']
        counts = []
        for option in options:
            key = str(words[index-1] + ' ' + option)
            prevWordCount = 0
            m = K*len(wordSet)
            probOpt = 1 / float(len(wordSet))  # P(Wi)
            if words[index-1] in wordFreq:
                prevWordCount = wordFreq[words[index-1]]
            if key not in bigramFreq:
                counts.append(m*probOpt/float(prevWordCount + m))
            else:
                counts.append(
                    (bigramFreq[key]+m*probOpt)/float(prevWordCount + m))
        counts = np.array(counts)
        pred = np.argmax(counts)
        totalPred += 1
        lineToAdd += " Prediction: "
        lineToAdd += options[pred]
        if options[pred] == validationData[i]['answer']:
            correctPred += 1
        f.write(lineToAdd + os.linesep)
    print(f"Accuracy on validataion set: {correctPred * 100 / totalPred}%")

K = 0.00001
correctPred = 0
totalPred = 0
outFile = './output.txt'
with open(outFile, 'a') as f:
    f.write(os.linesep + "Bigram Language Model - Add-k Smoothing With using the formula m = k*V, assuming V is the total number of words in training data" + os.linesep)
    for i in tqdm(range(len(validationData))):
        lineToAdd = "Question: "
        lineToAdd += validationData[i]['question']
        words = validationData[i]['question'].split()
        words = ['<start>'] + words + ['<end>']
        words = [word for word in words if word not in removeSet]
        index = words.index('XXXXX')
        words[index-1]
        options = validationData[i]['options']
        counts = []
        for option in options:
            key = str(words[index-1] + ' ' + option)
            prevWordCount = 0
            m = K*len(wordSet)
            probOpt = wordFreq.get(option, 0) / float(totalWords)  # P(Wi)
            if words[index-1] in wordFreq:
                prevWordCount = wordFreq[words[index-1]]
            if key not in bigramFreq:
                counts.append(m*probOpt/float(prevWordCount + m))
            else:
                counts.append(
                    (bigramFreq[key]+m*probOpt)/float(prevWordCount + m))
        counts = np.array(counts)
        pred = np.argmax(counts)
        totalPred += 1
        lineToAdd += " Prediction: "
        lineToAdd += options[pred]
        if options[pred] == validationData[i]['answer']:
            correctPred += 1
        f.write(lineToAdd + os.linesep)
    print(f"Accuracy on validataion set: {correctPred * 100 / totalPred}%")
