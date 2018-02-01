from unittest import TestCase, main

from sklearn import feature_extraction

from feature_extractor import FeatureExtractor
from tests import df, stopwords


class TestFeatureExtractor(TestCase):
    def test_fit(self):
        tfidf = feature_extraction.text.TfidfVectorizer(norm='l2', min_df=0, max_df=1, use_idf=True, smooth_idf=True,
                                                        sublinear_tf=True, stop_words=stopwords, analyzer='word',
                                                        strip_accents='unicode', lowercase=True, decode_error='ignore')
        fe = FeatureExtractor()
        tfidf.fit(df.stemmed)
        fe.fit(df['Intitul de la demande'])
        self.assertEqual(sorted(tfidf.vocabulary_), sorted(fe.vocabulary_))

    def test_fit_transform(self):
        self.fail()

    def test_transform(self):
        self.fail()

    def test_build_tokenizer(self):
        self.fail()


if __name__ == '__main__':
    main()
