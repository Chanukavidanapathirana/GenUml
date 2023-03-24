import sys
sys.path.append('serverside')
import os
import pickle
import unittest
from unittest.mock import MagicMock, patch
from Machine_Learning_Model.mainMLGenerate import getFinalDictionary, createUcArray, createNotUcList, generateFinalDictionary


class TestMyModule(unittest.TestCase):

    def test_createUcArray(self):
        # Mock the actorUcDictionary
        actorUcDictionary = {'actor1': ['uc1', 'uc2'], 'actor2': ['uc3']}
        
        # Call the function and check if it returns the expected result
        expected_result = ['uc1', 'uc2', 'uc3']
        self.assertEqual(createUcArray(actorUcDictionary), expected_result)

    def test_createNotUcList(self):
        # Mock the input values
        usecases = ['uc1', 'uc2', 'uc3']
        resultList = ['Use case', 'Not Use case', 'Use case']
        
        # Call the function and check if it returns the expected result
        expected_result = ['uc2']
        self.assertEqual(createNotUcList(usecases, resultList), expected_result)

    def test_generateFinalDictionary(self):
        # Mock the input values
        actorUcDictionary = {'actor1': ['uc1', 'uc2'], 'actor2': ['uc2', 'uc3']}
        notUcList = ['uc2']
        
        # Call the function and check if it returns the expected result
        expected_result = {'actor1': ['uc1'], 'actor2': ['uc3']}
        self.assertEqual(generateFinalDictionary(actorUcDictionary, notUcList), expected_result)

    @patch('Machine_Learning_Model.mainMLGenerate.pickle.load')
    def test_getFinalDictionary(self, mock_load):
        model_path = os.path.join('serverside/Machine_Learning_Model', 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Mock the necessary values
        actorUcDictionary = {'actor1': ['uc1', 'uc2'], 'actor2': ['uc3']}
        model = MagicMock()
        usecases = ['uc1', 'uc2', 'uc3']
        resultList = ['Use case', 'Not Use case', 'Use case']
        notUcList = ['uc2']
        
        # Configure the mock objects
        mock_load.return_value = model
        ML_classifiaction = MagicMock(return_value=resultList)
        
        # Call the function and check if it returns the expected result
        with patch('Machine_Learning_Model.mainMLGenerate.ML_classifiaction', ML_classifiaction):
            expected_result = {'actor1': ['uc1'], 'actor2': ['uc3']}
            self.assertEqual(getFinalDictionary(actorUcDictionary), expected_result)

if __name__ == '__main__':
    unittest.main()

