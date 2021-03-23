'''
empty
'''
import unittest

class SimpleTest(unittest.TestCase):
    '''
    simple case
    '''
    def setUp(self):
        '''
        setup
        '''
        print('setup')

    def test_simple(self):
        '''
        a simple test
        '''
        self.assertEqual(True, 1 == 1)


if __name__ == "__main__":
    unittest.main()
