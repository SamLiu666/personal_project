#!/usr/bin/env python
# coding: utf-8
import numpy as np
from collections import Counter
import itertools
from visual import show_tfidf


def get_idf(i2v, method="log"):

    """
    i2v: 字典，index-to-word
    methods: log, prob, len_norm
    如果IDF值约小，就说明它越没有意义，在其他文档中出现频率也很高。
    最后出来的IDF表是一个所有词语的重要程度表，
    return: idf矩阵 shape=[n_vocab, 1].
    """
    idf_methods = {
        "log": lambda x: 1 + np.log(len(docs) / (x+1)),
        "prob": lambda x: np.maximum(0, np.log((len(docs) - x) / (x+1))),
        "len_norm": lambda x: x / (np.sum(np.square(x))+1),
    }
    df = np.zeros((len(i2v),1))  # IDF 存储格式： （len_vocab, 1）
    for i in range(len(i2v)):
        # 统计词在文本中出现的次数
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df[i, 0] = d_count
    
    # 选择计算方式
    idf_function = idf_methods.get(method, None)
#     idf_function = idf_methods.get(method)
    if idf_function is None:
        raise ValueError
    return idf_function(df)


def get_tf(i2v, v2i, method="log"):
    """
    i2v: 字典，index-to-word
    v2i：字典，word-to-index
    类似get_idf：
    return：tf 矩阵，shape=[n_vocab, n_doc]
    """
    tf_methods = {
        "log": lambda x: np.log(1+x),
        "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
        "boolean": lambda x: np.minimum(x, 1),
        "log_avg": lambda x: (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis=1, keepdims=True))),
    }
    
    tf = np.zeros((len(i2v), len(docs)), dtype=np.float64)  # 
    for i,d in enumerate(docs_words):
        counter = Counter(d)  # 统计每个词出现的次数
        for v in counter.keys():
            # 单词出现次数，除以出现最多的次数
            tf[v2i[v], i] = counter[v] / counter.most_common(1)[0][1]
    
    tf_function = tf_methods.get(method, None)
    if tf_function is None:
        raise ValueError
    return tf_function(tf)


def get_keywords(tf_idf,i2v, n=2, len_docs=3):
    """获取文本的一些关键词"""
    key_words = []
    for c in range(len_docs):
        col = tf_idf[:, c]
        idx = np.argsort(col)[-n:]  # 逆序排列
        
        key_ = [i2v[i] for i in idx]
        print("doc{}, top{} keywords {}".format(c, n, key_))
        key_words.append(key_)
    return key_words


def cosine_similarity(q, _tf_idf):
    """查询句子和已有文本的余弦相似度： 
    q:待查询句子, tf_score 矩阵
    _tf_idf: 已有的tf_idf 矩阵"""
    unit_q = q / np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity


def docs_score(q, v2i,idf, tf_idf, len_norm=False):
    """文本分数计算
    q：查询文本，词列表
    i2v: 字典，index-to-word
    v2i：字典，word-to-index
    len_norm： 是否正则化 False
    """
    q_words = q.replace(",", "").split(" ")

    # add unknown words
    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i)-1] = v
            unknown_v += 1
    if unknown_v > 0:
        _idf = np.concatenate((idf, np.zeros((unknown_v, 1), dtype=np.float)), axis=0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype=np.float)), axis=0)
    else:
        _idf, _tf_idf = idf, tf_idf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=np.float)     # [n_vocab, 1]
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]

    q_vec = q_tf * _idf            # [n_vocab, 1]

    q_scores = cosine_similarity(q_vec, _tf_idf)
    if len_norm:
        # 长度正则化
        len_docs = [len(d) for d in docs_words]
        q_scores = q_scores / np.array(len_docs)
    return q_scores


def use_sklearn_tool(docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from visual import show_tfidf

    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(docs)
    print("idf: ", [(n, idf) for idf, n in zip(vectorizer.idf_, vectorizer.get_feature_names())])
    print("v2i: ", vectorizer.vocabulary_)

    q = "I get a coffee cup"
    qtf_idf = vectorizer.transform([q])
    res = cosine_similarity(tf_idf, qtf_idf)
    res = res.ravel().argsort()[-3:]
    print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in res[::-1]]))



if __name__ == '__main__':
    # 总共的文本
    docs = [
        "it is a good day, I like to stay here",
        "I am happy to be here",
        "I am bob",
        "it is sunny today",
        "I have a party today",
        "it is a dog and that is a cat",
        "there are dog and cat on the tree",
        "I study hard this morning",
        "today is a good day",
        "tomorrow will be a good day",
        "I like coffee, I like book and I like apple",
        "I do not like it",
        "I am kitty, I like bob",
        "I do not care who like bob, but I like kitty",
        "It is coffee time, bring your cup",
    ]


    # In[130]:


    docs_words = [d.replace(",", "").split(" ") for d in docs]  # 去处标点符号
    #vocab = set(itertools.chain(*docs_words))
    vocab = set()
    for sent in docs_words:
        for s in sent:
            vocab.add(s)
    # 构造词典
    v2i = {v:i for i,v in enumerate(vocab)}
    i2v = {i:v for v,i in v2i.items()}


    idf = get_idf(i2v)      # [n_vocab, 1]
    tf = get_tf(i2v, v2i)  # [n_vocab, n_doc]
    tf_idf = tf * idf      # [n_vocab, n_doc]
    key_word = get_keywords(tf_idf,i2v)
    print(key_word)


    print("tf shape(vecb in each docs): ", tf.shape)
    print("\ntf samples:\n", tf[:2])
    print("\nidf shape(vecb in all docs): ", idf.shape)
    print("\nidf samples:\n", idf[:2])
    print("\ntf_idf shape: ", tf_idf.shape)
    print("\ntf_idf sample:\n", tf_idf[:2])


    q = "I get a coffee cup"
    scores = docs_score(q, v2i,idf, tf_idf, len_norm=False)
    d_ids = scores.argsort()[-3:][::-1]
    print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in d_ids]))

    #show_tfidf(tf_idf.T, [i2v[i] for i in range(len(i2v))], "tfidf_matrix")

    print("#"*40)
    use_sklearn_tool(docs)