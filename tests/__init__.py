import string

import nltk
from pandas import read_csv

from feature_extractor import stopword_loader

# Loading data and stripping accents
url = "https://raw.githubusercontent.com/mkobbi/subvention-status-datacamp/master/data/subventions-accordees-et-refusees.csv"
df = read_csv(url, sep=";")
df = df.rename(columns=lambda x: x.decode('utf-8').encode('ascii', errors='ignore'))
df = df.fillna(value=0, axis='columns')
# Generating the unit test data set
df = df.sample(n=25, random_state=42)
# print(np.ravel(data['Intitul de la demande']))
# Declaration of necessary variables
stopwords = stopword_loader()
punctuation = set(string.punctuation)
# Old method implementation
string_columns = ['Nom du partenaire', "Intitul de la demande"]
data_old = df[string_columns].apply(
    lambda x: x.str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
stemmer = nltk.stem.snowball.FrenchStemmer()
df['stemmed'] = data_old['Intitul de la demande'].map(
    lambda x: ' '.join(list((filter(lambda y: y not in stopwords and y not in punctuation,
                                    [stemmer.stem(t) for t in set(t for t in
                                                                  nltk.word_tokenize(x, language='french',
                                                                                     preserve_line=False))
                                     if t.isalpha()])))))
