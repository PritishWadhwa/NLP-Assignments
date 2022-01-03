from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import re
from nltk.tokenize.casual import TweetTokenizer
df = pd.read_csv('./a01_spam.csv')
messageList = df['Message'].tolist()

tt = TweetTokenizer()
numEmoticons = 0
emoticonSet = set()
emoticonList = []
for i in messageList:
    words = tt.tokenize(i)
    for word in words:
        if re.search(
                r"^\:\w+\:$|^<[\\\/]?3$|^[\)\(\$\*][-^=]?[:;=]$|^[:;=][-^=]?[\(\)Pp0Oo3D\|\\\/\$\*\@$]", word):
            numEmoticons += 1
            emoticonSet.add(word)
            emoticonList.append(word)

for emoticon in emoticonList:
    print(emoticon)
print(numEmoticons)
