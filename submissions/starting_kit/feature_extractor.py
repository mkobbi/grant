# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import string
import unicodedata
import urllib

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


def stopword_loader(
        url="https://raw.githubusercontent.com/mkobbi/subvention-status-datacamp/master/data/stopwords-filter-fr.txt"):
    try:
        stopwords = str(urllib.urlopen(url).read().decode("utf-8").lower())
        stopwords = set(stopwords.split('\n'))
    except IOError:
        print('Failed to open "%s".', url)
    return stopwords


def document_preprocessor(doc):
    """ A custom document preprocessor

    This function can be edited to add some additional
    transformation on the documents prior to tokenization.

    """
    try:
        doc = unicode(doc, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass
    doc = unicodedata.normalize('NFKD', doc)
    doc = doc.encode('ascii', 'ignore')
    doc = doc.decode("utf-8")
    return str(doc).lower()


# def generate_tokens


def token_processor(sentence):
    """ A custom token processor

    This function can be edited to add some additional
    transformation on the extracted tokens (e.g. stemming)

    At present, this function just passes the tokens through.
    """
    stopwords = stopword_loader()
    punctuation = set(string.punctuation)
    punctuation.update(["``", "`", "..."])
    stemmer = nltk.stem.snowball.FrenchStemmer()
    stemmed_tokens = list((filter(lambda x: x not in stopwords and x not in punctuation,
                                  [stemmer.stem(t)
                                   for t in nltk.word_tokenize(sentence, 'french', False)
                                   if t.isalpha()])))
    for t in stemmed_tokens:
        yield t


class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw docs to a matrix of TF-IDF features. """

    def __init__(self):
        # see ``TfidfVectorizer`` documentation for other feature
        # extraction parameters.
        super(FeatureExtractor, self).__init__(norm='l2', min_df=0, max_df=1, use_idf=True, smooth_idf=True,
                                               lowercase=True, sublinear_tf=True, strip_accents='unicode',
                                               stop_words=stopword_loader(), analyzer='word', decode_error='ignore')

    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        X_df : pandas.DataFrame
            a DataFrame, where the text data is stored in the ``Intitul de la demande``
            column.
        """
        super(FeatureExtractor, self).fit(X_df)
        return self

    def fit_transform(self, X_df, y=None):
        self.fit(X_df)
        return self.transform(X_df)

    def transform(self, X_df):
        return super(FeatureExtractor, self).transform(X_df)

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        return lambda doc: token_processor(doc)
