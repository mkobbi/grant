# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import string
import unicodedata
import urllib

import nltk
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def stopword_loader(
        url="https://raw.githubusercontent.com/mkobbi/subvention-status-datacamp/master/data/stopwords-filter-fr.txt"):
    try:
        stopwords = str(urllib.urlopen(url).read().decode("utf-8").lower())
        stopwords = set(stopwords.split('\n'))
        return stopwords
    except IOError:
        print('Failed to open "%s".', url)


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


def cleanDataset(data):
    fr_stopwords_url = "https://raw.githubusercontent.com/mkobbi/subvention-status-datacamp/master/data/stopwords-filter-fr.txt"
    string_columns = ['Nom du partenaire']  # ["Intitul de la demande"]
    to_drop_columns = ["Anne", "Siret", "N SIMPA", 'CP-Adresse-Libell voie', "CP-Adresse-Ville"]
    str_categorical_columns = ["Nom du partenaire", "Appel  projets", "Appel  projets PolVille"]
    num_categorical_columns = ["Anne", "CP-Adresse-Code postal"]
    num_categorical_columns = ["CP-Adresse-Code postal"]
    data = data.fillna(value=0, axis='columns')
    data[string_columns] = data[string_columns].apply(
        lambda x: x.str.upper().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
    data[str_categorical_columns] = data[str_categorical_columns].apply(lambda x: x.astype('category').cat.codes)
    data[num_categorical_columns] = data[num_categorical_columns].apply(
        lambda x: x.astype('int'))  # .astype('category'))
    fr_stopwords = urllib.urlopen(fr_stopwords_url).read().decode("utf-8").upper()
    fr_stopwords = fr_stopwords.split('\n')
    # y = np.ravel(pd.DataFrame([data['Total vot'] > 0.0]).astype(int))
    # y = np.ravel(data.pop('y').values)
    data = data.drop(to_drop_columns, axis='columns')
    return data


class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw docs to a matrix of TF-IDF features. """

    def __init__(self):
        # see ``TfidfVectorizer`` documentation for other feature
        # extraction parameters.
        super(FeatureExtractor, self).__init__(strip_accents='unicode',
                                               stop_words=stopword_loader(), analyzer='word')

    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        X_df : pandas.DataFrame
            a DataFrame, where the text data is stored in the ``Intitul de la demande``
            column.
        """
        X_df = cleanDataset(X_df)
        super(FeatureExtractor, self).fit(X_df)
        return self

    def fit_transform(self, X_df, y=None):
        self.fit(X_df)
        return self.transform(X_df)

    def transform(self, X_df):
        # print "transform X_df="
        # print X_df
        # print "transform new="
        X_df = cleanDataset(X_df)
        words = super(FeatureExtractor, self).transform(X_df['Intitul de la demande'])

        X_df = X_df.drop(['Intitul de la demande'], axis='columns')
        data_sparse = scipy.sparse.csr_matrix(X_df.values[:, 1:])
        X = normalize(scipy.sparse.hstack((data_sparse, words)))
        return X

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: token_processor(doc)