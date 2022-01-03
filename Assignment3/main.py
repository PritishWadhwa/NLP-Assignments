from tqdm import tqdm
from nltk import bigrams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import copy


def split_tag_word(inp):
    """
    Returns word, tag for the given input
    """
    arr = inp.split('/')
    tag = arr[-1]
    del arr[-1]
    word = '/'.join(arr)
    return word, tag


def create_params(train_set):
    """
    Returns a dictionary of parameters
    tokens = Tokens present in the vocabulary
    tags = Tags present in the vocabulary
    token_tag = Count of each word/tag pair for emmission probability
    tag_pair = Count of each tag pair for transition probability
    vocabulary = Vocabulary of the training set
    Create params for HMM
    """
    tagged_sentences = []
    for sent in train_set:
        tagged_sentences.append(sent.split(' '))

    # list of list of tokens in the training set. Each internal set denotes a sentence.
    tokens = []
    tag_freq = {}  # list of all tags
    token_tag = {}  # dictionary of token/tag
    tag_pair = {}  # dictionary of tag tag
    vocab = set()

    # Create vocab, tokens, tags, token_tag
    for sentence in tagged_sentences:
        sentence_tokens = []  # all tokens for a sentence
        for word in sentence:
            if word == "":
                continue
            token_tag[word] = token_tag.get(word, 0) + 1
            actual_word, tag = split_tag_word(word)
            tag_freq[tag] = tag_freq.get(tag, 0) + 1
            vocab.add(actual_word)
            sentence_tokens.append(actual_word)
        tokens.append(sentence_tokens)

    # Create tag_pair
    for sentence in tagged_sentences:
        tag_bigram = bigrams(sentence)
        for t in tag_bigram:
            if t[0] == '' or t[1] == '':
                continue
            tag_1 = split_tag_word(t[0])[1]
            tag_2 = split_tag_word(t[1])[1]
            key = tag_1 + ' ' + tag_2
            tag_pair[key] = tag_pair.get(key, 0) + 1

    print("Vocabulary Length: ", len(vocab))
    print("Number of Tags: ", len(tag_freq.keys()))
    print(tag_freq)
    print("Number of Token/Tag: ", len(token_tag.keys()))
    print("Number of Tag Tag: ", len(tag_pair.keys()))
    return {
        "vocab": vocab,
        "tokens": tokens,
        "tag_freq": tag_freq,
        "token_tag": token_tag,
        "tag_pair": tag_pair
    }


def viterbi(params, sentence):
    """
    Takes a test sentence and returns predictions
    """
    vocab = params['vocab']
    tag_freq = params['tag_freq']
    tags = list(tag_freq.keys())
    token_tag = params['token_tag']
    tag_pair = params['tag_pair']
    prev_tag = '.'
    chosen_tags = []

    for word in sentence:
        count_emiss = True

        if word == "" or word not in vocab:
            count_emiss = False
            # chosen_tags.append('.')
            # In case of OOV words, choose the tag with the highest probability, Basically only the transition probability
            # print("Invalid token found: ", word)
            # continue
        probs = []

        for tag in tags:
            # Transmission probability
            p_trans = tag_pair.get(prev_tag + ' ' + tag,
                                   0) / tag_freq.get(prev_tag, -1)
            if count_emiss:
                # Emmission probability (only counted if the word is not OOV)
                p_emiss = token_tag.get(
                    word + "/" + tag, 0) / tag_freq.get(tag, -1)
                probs.append(p_trans * p_emiss)
            else:
                probs.append(p_trans)

        chosen_tag = tags[np.argmax(probs)]
        prev_tag = chosen_tag
        chosen_tags.append(chosen_tag)

    return chosen_tags


def HMM(train, test):
    """
    Takes trainset and test sentences and returns predictions
    """
    params = create_params(train)

    preds = []
    for i in tqdm(range(len(test))):
        sentence = test[i]
        predicted_tags = viterbi(params, sentence)
        preds.append(predicted_tags)

    tag_freq = params['tag_freq']
    tags = list(tag_freq.keys())

    return tags, preds


def evaluate_HMM(tags, preds, test_y, to_print=True):
    """
    tags: List of tags
    preds: List of list of predicted tags
    """
    # for i in range(len(test_y)):
    #     for j in range(len(test_y[i])):

    flat_tags = [item for elem in test_y for item in elem]
    flat_preds = [item for elem in preds for item in elem]

    accuracy = accuracy_score(flat_tags, flat_preds)
    precision = precision_score(flat_tags, flat_preds, average='weighted')
    recall = recall_score(flat_tags, flat_preds, average='weighted')
    f1_score = (2 * precision * recall) / (precision + recall)

    tagwise_precision = precision_score(
        flat_tags, flat_preds, labels=tags, pos_label=None, average=None)
    tagwise_recall = recall_score(
        flat_tags, flat_preds, labels=tags, pos_label=None, average=None)
    tagwise_f1 = (2 * tagwise_precision * tagwise_recall) / \
        (tagwise_precision + tagwise_recall)

    cm = confusion_matrix(flat_tags, flat_preds)

    # cm = np.zeros((len(tags), len(tags)))

    # for sent_y, sent_pred in zip(test_y, preds):
    #     for tag, pred in zip(sent_y, sent_pred):
    #         cm[tags.index(tag)][tags.index(pred)] += 1

    if to_print:
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score", f1_score)
        print("Tagwise Precision: ", tagwise_precision)
        print("Tagwise Recall: ", tagwise_recall)
        print("Tagwise F1 score: ", tagwise_f1)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tags)
        disp.plot()
        plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1_score,
        "cm": cm,
        "tagwise_precision": tagwise_precision,
        "tagwise_recall": tagwise_recall,
        "tagwise_f1": tagwise_f1
    }


def load_data(filename):
    """
    Loads the data from the given file
    """
    with open(filename, 'r') as f:
        data = f.read().splitlines()
    return np.array(data)


def split_Xy(test_Xy):
    """
    test_Xy: List of list of tokens and tags
    Returns: List of tokens and list of tags
    """
    test_y = []
    test_X = []

    for sent in test_Xy:
        tagged_sent = sent.split(' ')
        sent_y = []
        sent_X = []

        for word in tagged_sent:
            if word == "":
                continue
            actual_word, tag = split_tag_word(word)
            sent_X.append(actual_word)
            sent_y.append(tag)

        test_y.append(sent_y)
        test_X.append(sent_X)

    return test_X, test_y


def main():
    data = load_data('Brown_tagged_train.txt')
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    unique_tags = []
    metrics = []
    for train_index, test_index in kf.split(data):
        train_set, test_Xy = data[train_index], data[test_index]
        test_X, test_y = split_Xy(test_Xy)
        tags, preds = HMM(train_set, test_X)
        unique_tags = tags
        metrics.append(evaluate_HMM(tags, preds, test_y, to_print=False))

    print("Average metrics: ")
    precision, recall, f1 = 0, 0, 0
    tagwise_precision, tagwise_recall, tagwise_f1 = np.zeros(
        len(unique_tags)), np.zeros(len(unique_tags)), np.zeros(len(unique_tags))

    cm = np.zeros((len(unique_tags), len(unique_tags)))

    for metric_dict in metrics:
        precision += metric_dict["precision"]
        recall += metric_dict["recall"]
        f1 += metric_dict["f1-score"]
        tagwise_precision += metric_dict["tagwise_precision"]
        tagwise_recall += metric_dict["tagwise_recall"]
        tagwise_f1 += metric_dict["tagwise_f1"]
        cm += metric_dict["cm"]

    precision /= len(metrics)
    recall /= len(metrics)
    f1 /= len(metrics)
    tagwise_precision /= len(metrics)
    tagwise_recall /= len(metrics)
    tagwise_f1 /= len(metrics)

    # cm /= len(metrics)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1)
    # create df of tagwise metrics
    tagwise_metrics = pd.DataFrame(
        {
            "tag": unique_tags,
            "precision": tagwise_precision,
            "recall": tagwise_recall,
            "f1-score": tagwise_f1
        }
    )
    print(tagwise_metrics)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=unique_tags)
    disp.plot()
    plt.show()


main()
