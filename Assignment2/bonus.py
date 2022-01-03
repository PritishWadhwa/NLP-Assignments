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
    f.write(os.linesep + "Bonus Part - Without Smoothing" + os.linesep)
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
            probPrev = 1.0
            probNext = 1.0
            keyPrev = str(words[index-1] + ' ' + option)
            keyNext = str(option + ' ' + words[index+1])
            if keyPrev not in bigramFreq:
                probPrev = 0.0
            else:
                probPrev = bigramFreq[keyPrev]/float(wordFreq[words[index-1]])
            if keyNext not in bigramFreq:
                probNext = 0.0
            else:
                probNext = bigramFreq[keyNext]/float(wordFreq[option])
            counts.append(probPrev * probNext)
        counts = np.array(counts)
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
    f.write(os.linesep + "Bonus Part - Add-1 Smoothing" + os.linesep)
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
            probPrev = 1.0
            probNext = 1.0
            keyPrev = str(words[index-1] + ' ' + option)
            keyNext = str(option + ' ' + words[index+1])
            if keyPrev not in bigramFreq:
                probPrev = 1.0/float(len(wordSet))
            else:
                probPrev = (bigramFreq[keyPrev] + 1.0) / \
                    float(wordFreq[words[index-1]] + len(wordSet))
            if keyNext not in bigramFreq:
                probNext = 1.0/float(len(wordSet))
            else:
                probNext = (bigramFreq[keyNext] + 1.0) / \
                    float(wordFreq[option] + len(wordSet))
            counts.append(probPrev * probNext)
        counts = np.array(counts)
        pred = np.argmax(counts)
        totalPred += 1
        lineToAdd += " Prediction: "
        lineToAdd += options[pred]
        if options[pred] == validationData[i]['answer']:
            correctPred += 1
        f.write(lineToAdd + os.linesep)
    print(f"Accuracy on validataion set: {correctPred * 100 / totalPred}%")

K = 0.001
correctPred = 0
totalPred = 0
outFile = './output.txt'
with open(outFile, 'a') as f:
    f.write(os.linesep + "Bonus Part - Add-k Smoothing Without using the formula m = k*V" + os.linesep)
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
            probPrev = 1.0
            probNext = 1.0
            keyPrev = str(words[index-1] + ' ' + option)
            keyNext = str(option + ' ' + words[index+1])
            if keyPrev not in bigramFreq:
                probPrev = K/float(K*len(wordSet))
            else:
                probPrev = (bigramFreq[keyPrev] + K) / \
                    float(wordFreq[words[index-1]] + K*len(wordSet))
            if keyNext not in bigramFreq:
                probNext = K/float(K*len(wordSet))
            else:
                probNext = (bigramFreq[keyNext] + K) / \
                    float(wordFreq[option] + K*len(wordSet))
            counts.append(probPrev * probNext)
        counts = np.array(counts)
        pred = np.argmax(counts)
        totalPred += 1
        lineToAdd += " Prediction: "
        lineToAdd += options[pred]
        if options[pred] == validationData[i]['answer']:
            correctPred += 1
        f.write(lineToAdd + os.linesep)
    print(f"Accuracy on validataion set: {correctPred * 100 / totalPred}%")

K = 0.00000001
correctPred = 0
totalPred = 0
outFile = './output.txt'
with open(outFile, 'a') as f:
    f.write(os.linesep + "Bonus Part - Add-k Smoothing With using the formula m = k*V, assuming V is the number of distinct words in vocabulary" + os.linesep)
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
            probPrev = 1.0
            probNext = 1.0
            keyPrev = str(words[index-1] + ' ' + option)
            keyNext = str(option + ' ' + words[index+1])
            m = K*len(wordSet)
            probOpt = 1 / float(len(wordSet))  # P(Wi)
            if keyPrev not in bigramFreq:
                probPrev = (m*probOpt) / \
                    float(m + wordFreq.get(words[index-1], 0))
            else:
                probPrev = (bigramFreq[keyPrev] + (m*probOpt)) / \
                    float(m + wordFreq.get(words[index-1], 0))
            if keyNext not in bigramFreq:
                probPrev = (m*probOpt)/float(m + wordFreq.get(option, 0))
            else:
                probNext = (bigramFreq[keyNext] + (m*probOpt)) / \
                    float(m + wordFreq.get(option, 0))
            counts.append(probPrev * probNext)
        counts = np.array(counts)
        pred = np.argmax(counts)
        totalPred += 1
        lineToAdd += " Prediction: "
        lineToAdd += options[pred]
        if options[pred] == validationData[i]['answer']:
            correctPred += 1
        f.write(lineToAdd + os.linesep)
    print(f"Accuracy on validataion set: {correctPred * 100 / totalPred}%")

K = 0.000001
correctPred = 0
totalPred = 0
outFile = './output.txt'
with open(outFile, 'a') as f:
    f.write(os.linesep + "Bonus Part - Add-k Smoothing With using the formula m = k*V, assuming V is the total number of words in the vocabulary" + os.linesep)
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
            probPrev = 1.0
            probNext = 1.0
            keyPrev = str(words[index-1] + ' ' + option)
            keyNext = str(option + ' ' + words[index+1])
            m = K*len(wordSet)
            probOpt = wordFreq.get(option, 0) / float(totalWords)  # P(Wi)
            if keyPrev not in bigramFreq:
                probPrev = (m*probOpt) / \
                    float(m + wordFreq.get(words[index-1], 0))
            else:
                probPrev = (bigramFreq[keyPrev] + (m*probOpt))/float(m +
                                                                     wordFreq.get(words[index-1], 0))
            if keyNext not in bigramFreq:
                probPrev = (m*probOpt)/float(m +
                                             wordFreq.get(option, 0))
            else:
                probNext = (bigramFreq[keyNext] + (m*probOpt))/float(m +
                                                                     wordFreq.get(option, 0))
            counts.append(probPrev * probNext)
        counts = np.array(counts)
        pred = np.argmax(counts)
        totalPred += 1
        lineToAdd += " Prediction: "
        lineToAdd += options[pred]
        if options[pred] == validationData[i]['answer']:
            correctPred += 1
        f.write(lineToAdd + os.linesep)
    print(f"Accuracy on validataion set: {correctPred * 100 / totalPred}%")
