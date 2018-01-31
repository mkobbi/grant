# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import urllib

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


def stopword_loader(
        url="https://raw.githubusercontent.com/mkobbi/subvention-status-datacamp/master/data/stopwords-filter-fr.txt"):
    try:
        stopwords = urllib.urlopen(url).read().decode("utf-8").upper()
        stopwords = stopwords.split('\n')
    except IOError:
        print('Failed to open "%s".', url)
    return stopwords


def document_preprocessor(sentence):
    """ A custom document preprocessor

    This function can be edited to add some additional
    transformation on the documents prior to tokenization.

    """
    stopwords = stopword_loader()
    stemmer = nltk.stem.snowball.FrenchStemmer()

    return sentence


def token_processor(tokens):
    """ A custom token processor

    This function can be edited to add some additional
    transformation on the extracted tokens (e.g. stemming)

    At present, this function just passes the tokens through.
    """
    stemmer = nltk.stem.snowball.FrenchStemmer()
    return list((filter(lambda x: x.lower() not in stopwords and
                                  x.lower() not in punctuation,
                        [stemmer.stem(t.lower())
                         for t in word_tokenize(sentence)
                         if t.isalpha()])))

    for t in tokens:
        yield t


class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw docs to a matrix of TF-IDF features. """

    def __init__(self):
        # see ``TfidfVectorizer`` documentation for other feature
        # extraction parameters.
        super(FeatureExtractor, self).__init__(
                analyzer='word', preprocessor=document_preprocessor)

    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        X_df : pandas.DataFrame
            a DataFrame, where the text data is stored in the ``statement``
            column.
        """
        X_df = X_df.rename(columns=lambda x: x.decode('utf-8').encode('ascii', errors='ignore'))
        string_columns = ["Nom du partenaire", 'Intitul de la demande']
        to_drop_columns = ["Anne", "Siret", "N SIMPA", 'CP-Adresse-Libell voie', "CP-Adresse-Ville"]
        str_categorical_columns = ["Nom du partenaire", "Appel  projets", "Appel  projets PolVille"]
        num_categorical_columns = ["Anne", "CP-Adresse-Code postal"]
        X_df = X_df.fillna(value=0, axis='columns')
        X_df[string_columns] = X_df[string_columns].apply(
            lambda x: x.str.upper().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
        X_df[str_categorical_columns] = X_df[str_categorical_columns].apply(lambda x: x.astype('category').cat.codes)
        X_df[num_categorical_columns] = X_df[num_categorical_columns].apply(lambda x: x.astype('int'))
        X_df = X_df.drop(to_drop_columns, axis='columns')
        super(FeatureExtractor, self).fit(X_df.statement)
        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):
        X = super(FeatureExtractor, self).transform(X_df.statement)
        return X

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))
