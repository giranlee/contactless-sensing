import re
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

dictionary = pd.concat([pd.read_excel('synonyms.xlsx'), pd.read_excel('special_words.xlsx')])
dictionary = dictionary.sort_values('order', kind='stable')

rules = []
for order, group in dictionary.groupby('order'):
    term = group['term'].iloc[0]
    pattern = '|'.join(group['pattern'])
    rules.append((term, re.compile(r'(?<![a-z0-9_])(' + pattern + r')(?![a-z0-9_])')))
terms = [term for term, _ in rules]

def remove_abbreviation(match):
    word = match.group(1)
    is_abbreviation = re.fullmatch(r'[A-Za-z0-9\-]+', word) and sum(c.isupper() for c in word) >= 2
    if is_abbreviation or word == 'mmWave':
        return ''
    return match.group(0)

def remove_repeated_terms(text):
    for term in terms:
        text = re.sub(r'\b' + term + r'(\s+' + term + r'\b)+', term, text)
    return text

def clean(text):
    # remove abbreviations in parentheses, e.g. "channel state information (CSI)"
    text = re.sub(r'\s*\(([^()]*)\)', remove_abbreviation, text)
    text = text.lower()

    # remove "e.g.", "i.e.", "et al." and possessive 's
    text = re.sub(r'\b(e\.g\.|i\.e\.|et al\.)', ' ', text)
    text = re.sub(r"['’]s\b", '', text)

    # synonyms and multi-word terms
    for term, pattern in rules:
        text = pattern.sub(term, text)
    # remove numbers (including percentages), symbols, and one-letter words
    text = re.sub(r'\d+(\.\d+)?\s*%', ' ', text)
    text = re.sub(r'\b\d+([.,]\d+)*\b', ' ', text)
    text = re.sub(r'[^a-z_\-\s.,;:]', ' ', text)
    text = re.sub(r'(?<![a-z_\-])[a-z](?![a-z_\-])', ' ', text)
    text = ' '.join(text.split())
    return remove_repeated_terms(text)

df = pd.read_excel('abstracts.xlsx')
df['Text_Clean'] = df['Abstract'].fillna('').apply(clean)
df[['Study_ID', 'Year', 'Title', 'Text_Clean']].to_excel('preprocessed_abstracts.xlsx', index=False)

keep = {'system', 'show', 'find', 'found', 'detail', 'fire', 'bill', 'interest', 'call', 'describe', 'move', 'front', 'back', 'full', 'side', 'top', 'bottom', 'part', 'amount', 'computer'}
stopwords = set(ENGLISH_STOP_WORDS) - keep

def tokenize(text):
    return re.findall(r'[a-z][a-z_\-]*[a-z]', text)

all_words = Counter()
for text in df['Text_Clean']:
    all_words.update(tokenize(text))

def singular(word):
    if '_' in word or word in stopwords or len(word) <= 3:
        return word
    if word.endswith('ies') and word[:-3] + 'y' in all_words:
        return word[:-3] + 'y'
    if word.endswith('s') and not word.endswith('ss') and word[:-1] in all_words:
        return word[:-1]
    return word

tf = Counter()
doc_count = Counter()
for text in df['Text_Clean']:
    words = [singular(w) for w in tokenize(text) if w not in stopwords]
    tf.update(words)
    doc_count.update(set(words))

freq = pd.DataFrame({'word': list(tf), 'TF': [tf[w] for w in tf], 'DF': [doc_count[w] for w in tf]})
freq = freq.sort_values(['DF', 'TF'], ascending=False)
freq.to_excel('word_frequency.xlsx', index=False)
print('documents:', len(df), '| unique words:', len(freq))
