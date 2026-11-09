import re
import collections
import numpy as np
import pandas as pd

STEP = 1

FILE_DATA = '분석데이터셋_비접촉무선센싱.xlsx'
FILE_CLEAN = '전처리_NetMiner입력_비접촉무선센싱.xlsx'
FILE_STOP = 'stopwords.xlsx'
FILE_WORDS = 'word_statistics_2.xlsx'
SEED = 42
EMBED_MODEL = 'all-mpnet-base-v2'
TOP_N = 10

EXTRA_STOP = ['older_adult', 'using', 'non', 'detect', 'lbrace', 'rbrace', 'esfv']

GRID = {'n_neighbors': [5, 10, 15], 'min_cluster_size': [5, 7, 10, 15], 'min_samples': [None, 3]}

FINAL = {'n_neighbors': 15, 'min_cluster_size': 15, 'min_samples': 3}

data = pd.read_excel(FILE_DATA, sheet_name='분석데이터')
clean = pd.read_excel(FILE_CLEAN, sheet_name='NetMiner입력')[['Study_ID', 'Text_Clean']]
data = data.merge(clean, on='Study_ID', how='left')
assert len(data) == 328 and data['Text_Clean'].notna().all(), 'check input files'
docs_embed = data['Abstract_For_Analysis'].fillna('').tolist()
years = data['출판연도'].astype(int).tolist()

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
user_stop = set(pd.read_excel(FILE_STOP)['word'].dropna().astype(str).str.strip())
STOP = set(ENGLISH_STOP_WORDS) | user_stop | set(EXTRA_STOP)

tok = lambda s: re.findall(r'[a-z][a-z_]*[a-z]', s.lower())
raw_vocab = collections.Counter(w for s in data['Text_Clean'] for w in tok(s))

def lemma(w):
    if '_' in w or len(w) <= 3:
        return w
    if w.endswith('ies') and w[:-3] + 'y' in raw_vocab:
        return w[:-3] + 'y'
    if w.endswith('s') and not w.endswith('ss') and w[:-1] in raw_vocab:
        return w[:-1]
    return w

tokens = [[lemma(w) for w in tok(s)] for s in data['Text_Clean']]
nouns = set(pd.read_excel(FILE_WORDS).iloc[:, 0].astype(str).str.strip())
tokens = [[w for w in t if w in nouns and w not in STOP and len(w) > 1] for t in tokens]
docs_repr = [' '.join(t) for t in tokens]

import os
from sentence_transformers import SentenceTransformer
EMB_FILE = 'embeddings_all-mpnet-base-v2.npy'
if os.path.exists(EMB_FILE):
    emb = np.load(EMB_FILE)
else:
    emb = SentenceTransformer(EMBED_MODEL).encode(docs_embed, show_progress_bar=True)
    np.save(EMB_FILE, emb)

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

def build(n_neighbors, min_cluster_size, min_samples):
    return BERTopic(
        embedding_model=EMBED_MODEL,
        umap_model=UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine', random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean',
                              cluster_selection_method='eom', prediction_data=True),
        vectorizer_model=CountVectorizer(token_pattern=r'(?u)\b[a-z][a-z_]*[a-z]\b'),
        top_n_words=TOP_N, calculate_probabilities=False)

def topic_words(model):
    return {k: [w for w, _ in v][:TOP_N] for k, v in model.get_topics().items() if k != -1}

def diversity(tw):
    allw = [w for ws in tw.values() for w in ws]
    return len(set(allw)) / len(allw) if allw else np.nan

def npmi_doc(tw):
    sets = [set(t) for t in tokens]; n = len(sets)
    df = collections.Counter(w for s in sets for w in s)
    vals = []
    for ws in tw.values():
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, b = ws[i], ws[j]
                pa, pb = df[a] / n, df[b] / n
                pab = sum(1 for s in sets if a in s and b in s) / n
                if pa == 0 or pb == 0:
                    continue
                vals.append(-1.0 if pab == 0 else (np.log(pab / (pa * pb)) / -np.log(pab) if pab < 1 else 1.0))
    return float(np.mean(vals)) if vals else np.nan

def _window_matrix(topics, size):
    vocab = sorted({w for t in topics for w in t}); ix = {w: i for i, w in enumerate(vocab)}
    rows = []
    for t in tokens:
        wins = [t] if len(t) <= size else [t[i:i + size] for i in range(len(t) - size + 1)]
        for win in wins:
            r = np.zeros(len(vocab), dtype=np.float32)
            for w in set(win):
                if w in ix: r[ix[w]] = 1
            rows.append(r)
    M = np.array(rows)
    return ix, M.mean(0), (M.T @ M) / len(M)

def _nlr(p12, pi, pj, eps=1e-12):
    return np.log((p12 + eps) / (pi * pj)) / -np.log(p12 + eps)

def coherence_own(tw, measure):
    topics = [ws for ws in tw.values() if len(ws) >= 2]
    size = 110 if measure == 'c_v' else 10
    ix, p1, p12 = _window_matrix(topics, size)
    scores = []
    for ws in topics:
        k = [ix[w] for w in ws]
        N = np.array([[_nlr(p12[a, b], p1[a], p1[b]) for b in k] for a in k])
        if measure == 'c_v':
            vW = N.sum(0)
            cos = [N[i] @ vW / (np.linalg.norm(N[i]) * np.linalg.norm(vW) + 1e-12) for i in range(len(k))]
            scores.append(np.mean(cos))
        else:
            scores.append(np.mean([N[i, j] for i in range(len(k)) for j in range(len(k)) if i != j]))
    return float(np.mean(scores))

try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    _gdict = Dictionary(tokens)
    COH_TOOL = 'gensim'

    def coherence(tw, measure):
        topics = [[w for w in ws if w in _gdict.token2id] for ws in tw.values()]
        topics = [t for t in topics if len(t) >= 2]
        return CoherenceModel(topics=topics, texts=tokens, dictionary=_gdict, coherence=measure, processes=1).get_coherence()
except ImportError:
    COH_TOOL = 'own'
    coherence = coherence_own
print('coherence:', COH_TOOL)

if STEP == 1:
    import itertools
    rows = []
    for nn, mcs, ms in itertools.product(GRID['n_neighbors'], GRID['min_cluster_size'], GRID['min_samples']):
        m = build(nn, mcs, ms)
        topics, _ = m.fit_transform(docs_repr, embeddings=emb)
        tw = topic_words(m)
        k = len(tw)
        row = {'n_neighbors': nn, 'min_cluster_size': mcs, 'min_samples': ms if ms is not None else 'None',
               'topics, n': k, 'outlier, n': int(sum(t == -1 for t in topics)),
               'outlier (%)': round(100 * np.mean(np.array(topics) == -1), 1),
               'diversity': round(diversity(tw), 3), 'NPMI (doc)': round(npmi_doc(tw), 3) if k else np.nan}
        if k >= 2:
            row['C_v'] = round(coherence(tw, 'c_v'), 3)
            row['NPMI'] = round(coherence(tw, 'c_npmi'), 3)
        rows.append(row)
        print(row)
    g = pd.DataFrame(rows)
    g.to_excel('bertopic_1_grid.xlsx', index=False)
    print('saved: bertopic_1_grid.xlsx')

if STEP == 2:
    m = build(FINAL['n_neighbors'], FINAL['min_cluster_size'], FINAL['min_samples'])
    topics, _ = m.fit_transform(docs_repr, embeddings=emb)
    tw = topic_words(m)
    info = m.get_topic_info()
    info['Top words'] = info['Topic'].map(lambda k: ', '.join(w for w, _ in m.get_topic(k)[:TOP_N]))
    info['Articles (%)'] = (info['Count'] / len(data) * 100).round(1)

    doc = data[['Study_ID', '출판연도', '제목']].rename(columns={'출판연도': 'Year', '제목': 'Title'})
    doc['Topic'] = topics
    doc.to_excel('bertopic_2_doc_topics.xlsx', index=False)

    rep = []
    for k, ds in m.get_representative_docs().items():
        for d in ds:
            i = docs_repr.index(d)
            rep.append({'Topic': k, 'Study_ID': data.at[i, 'Study_ID'], 'Year': years[i], 'Title': data.at[i, '제목'],
                        'Abstract': data.at[i, 'Abstract_For_Analysis']})
    ty = doc.pivot_table(index='Year', columns='Topic', values='Study_ID', aggfunc='count').fillna(0).astype(int)

    metrics = {'topics, n': len(tw), 'outlier, n': int(sum(t == -1 for t in topics)),
               'diversity': round(diversity(tw), 3), 'NPMI (doc)': round(npmi_doc(tw), 3)}
    if len(tw) >= 2:
        metrics['C_v'] = round(coherence(tw, 'c_v'), 3); metrics['NPMI'] = round(coherence(tw, 'c_npmi'), 3)

    with pd.ExcelWriter('bertopic_2_results.xlsx') as w:
        info[['Topic', 'Count', 'Articles (%)', 'Top words', 'Name']].to_excel(w, sheet_name='topics', index=False)
        pd.DataFrame(rep).to_excel(w, sheet_name='representative abstracts', index=False)
        ty.to_excel(w, sheet_name='topic by year')
        pd.DataFrame([{**FINAL, 'seed': SEED, 'embedding': EMBED_MODEL, **metrics}]).to_excel(w, sheet_name='model', index=False)
        try:
            from umap import UMAP as U2
            te = m.topic_embeddings_[1:] if -1 in m.get_topics() else m.topic_embeddings_
            xy = U2(n_neighbors=2, n_components=2, metric='cosine', random_state=SEED).fit_transform(te)
            pd.DataFrame({'Topic': sorted(tw), 'x': xy[:, 0], 'y': xy[:, 1],
                          'Count': [int((np.array(topics) == k).sum()) for k in sorted(tw)]}).to_excel(w, sheet_name='topic map xy', index=False)
        except Exception as e:
            print('topic map skipped:', e)
    m.visualize_topics().write_html('bertopic_2_intertopic_map.html')
    m.visualize_barchart(top_n_topics=len(tw), n_words=TOP_N).write_html('bertopic_2_barchart.html')
    print('saved: bertopic_2_results.xlsx, bertopic_2_doc_topics.xlsx')
    print(metrics)
