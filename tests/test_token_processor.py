from unittest import TestCase, main

import nltk

from feature_extractor import document_preprocessor
from tests import data


class TestToken_processor(TestCase):
    def test_token_processor(self):
        for sentence in data['Intitul de la demande']:
            tokens = [t for t in nltk.word_tokenize(document_preprocessor(sentence.lower()), language='french',
                                                    preserve_line=False)]
            print(tokens)
        self.assertEqual(tokens, nltk.word_tokenize(document_preprocessor(sentence.lower()), language='french',
                                                    preserve_line=False))


if __name__ == '__main__':
    main()
