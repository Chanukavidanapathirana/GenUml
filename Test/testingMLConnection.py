import unittest
import sys
sys.path.append('serverside')
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from Machine_Learning_Model.MLModelConnection import ML_classifiaction

class TestMLClassification(unittest.TestCase):

    def setUp(self):
        self.model = MultinomialNB()
        self.vectorizer = CountVectorizer()
        self.data = pd.read_csv('serverside/Machine_Learning_Model/DATASET2.csv')
        self.X_train = self.vectorizer.fit_transform(self.data['Scenario'])
        self.y_train = self.data['Classification']
        self.model.fit(self.X_train, self.y_train)

    def test_ML_classification(self):
        # Test for a single usecase
        usecase = ["User should be able to login to the website"]
        expected_output = ['Login']
        self.assertEqual(ML_classifiaction(usecase, self.model), expected_output)

        # Test for multiple usecases
        usecases = ["User should be able to create a new account",
                    "User should be able to reset their password"]
        expected_output = ['Registration', 'ForgotPassword']
        self.assertEqual(ML_classifiaction(usecases, self.model), expected_output)

if __name__ == '__main__':
    unittest.main()
