from django.test import TestCase
from iondb.security.models import SecureString

MY_STRING = 'nothing to see here, move along'
NAME='test'

class SecureStringTestCase(TestCase):
    """Test case for password"""

    def test_init(self):
        """make sure an init does not work"""
        with self.assertRaises(Exception):
            SecureString(unencrypted=MY_STRING, name=NAME)

        # unfortunately we cannot assure ourselves of manually assigned encryptions
        #with self.assertRaises(Exception):
        #    SecureString(encrypted_string=MY_PASSWORD, name=NAME)

    def test_save(self):
        """Setup the test cases"""
        SecureString.create(MY_STRING, NAME).save()

        with self.assertRaises(Exception):
            SecureString.create(name=NAME).save()

        with self.assertRaises(Exception):
            SecureString(encrypted_string='something', name=NAME).save()

    def test_create(self):
        """test the objects create method"""
        self.assertIsNotNone(SecureString.create(unencrypted=MY_STRING, name=NAME))

        with self.assertRaises(Exception):
            SecureString.objects.create(name=NAME)

        with self.assertRaises(TypeError):
            SecureString.objects.create(password=MY_STRING, name=NAME)

    def test_get(self):
        """tests the get method"""
        SecureString.create(MY_STRING, NAME).save()
        self.assertIsNotNone(SecureString.objects.get(name=NAME))

    def test_decrypt(self):
        """tests the get method"""

        SecureString.create(MY_STRING, NAME).save()
        sp = SecureString.objects.get(name=NAME)
        self.assertEqual(sp.decrypted, MY_STRING)

    def test_created(self):
        SecureString.create(MY_STRING, NAME).save()
        self.assertIsNotNone(SecureString.objects.get(name=NAME).created)
