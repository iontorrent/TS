# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf import settings
from requests.auth import HTTPBasicAuth
from tastypie.test import ResourceTestCase
import sys

HOST = 'localhost'
API_PATH = 'security/api/v1/securestring'
USER = 'ionadmin'
PASS = 'ionadmin'
AUTH = HTTPBasicAuth(USER, PASS)

class SecureStringResourceTest(ResourceTestCase):

    def setUp(self):
        """setup the tests"""

        # we need to remove the base context processor because it requires a global config
        settings.TEMPLATE_CONTEXT_PROCESSORS = [x for x in settings.TEMPLATE_CONTEXT_PROCESSORS if x != 'iondb.rundb.context_processors.base_context_processor']
        settings.DEBUG=True
        settings.LOGGING = {
                            'version': 1,
                            'disable_existing_loggers': False,
                            'formatters': {
                                'verbose': {
                                    'format': "[%(asctime)s] %(levelname)s %(message)s",
                                    'datefmt': "%d/%b/%Y %H:%M:%S"
                                }
                            },
                            'handlers': {
                                'file': {
                                    'level': 'DEBUG',
                                    'class': 'logging.FileHandler',
                                    'filename': '/var/log/django_practices.log',
                                    'formatter': 'verbose'
                                },
                                'console': {
                                    'level': 'DEBUG',
                                    'class': 'logging.StreamHandler',
                                    'stream': sys.stdout,
                                    'formatter': 'verbose'
                                },
                            },
                            'loggers': {

                                'django_test': {
                                    'handlers': ['file', 'console'],
                                    'level': 'DEBUG',
                                },
                                'iondb.security': {
                                    'handlers': ['file', 'console'],
                                    'level': 'DEBUG',
                                }

                            }
                        }
        super(SecureStringResourceTest, self).setUp()

    def get_credentials(self):
        return self.create_basic(username=USER, password=PASS)

    def test_get_list_unauthenticated(self):
        self.assertHttpUnauthorized(self.api_client.get('http://%s/%s/' % (HOST, API_PATH), format='json'))

    def test_get_list_json(self):
        #resp = requests.get('https://%s/%s/' % (HOST, API_PATH), verify=False, auth=AUTH)
        resp = self.api_client.get('https://%s/%s/' % (HOST, API_PATH), format='json', authentication=self.get_credentials())
        self.assertHttpOK(resp)
        self.assertValidJSON(resp.content)

    def test_post_item(self):
        SECURED_STRING = {'unencrypted': 'this is my secret', 'name': 'secret for testing'}

        resp = self.api_client.post('http://%s/%s/' % (HOST, API_PATH), format='json', data=SECURED_STRING, authentication=self.get_credentials(), headers={"content-type": "application/json"})
        self.assertHttpCreated(resp)