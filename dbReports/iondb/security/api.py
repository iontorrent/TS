# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
import logging
from iondb.rundb.authn import IonAuthentication
from iondb.security.models import SecureString
from tastypie import fields
from tastypie.http import HttpApplicationError
from tastypie.resources import ModelResource, ALL
from tastypie.authorization import DjangoAuthorization

# create a logger
logger = logging.getLogger(__name__)

class SecureStringResource(ModelResource):
    """Resource for distributing the secure password."""
    decrypted = fields.CharField(readonly=True, attribute='decrypted')

    def obj_create(self, bundle, request=None, **kwargs):
        """Custom create method to override the default post"""

        try:
            params = bundle.data
            # sanity check on the input parameters
            if 'encrypted_string' in params:
                del params['encrypted_string']
            if 'unencrypted' not in params:
                return HttpApplicationError("API creation of SecureString requires \'unencrypted\' parameter")
            if 'name' not in params:
                return HttpApplicationError("API creation of SecureString requires \'name\' parameter")

            # create and save the new object
            try:
                secured = SecureString.objects.get(name=params['name'])
                secured.encrypt(params['unencrypted'])
                secured.save()
            except SecureString.DoesNotExist:
                secured = SecureString.create(params['unencrypted'], params['name'])
                secured.save()

            return bundle
        except Exception as exc:
            logger.debug(str(exc))
            return HttpApplicationError("Error creating secure string: " + str(exc))

    class Meta:
        queryset = SecureString.objects.all()
        excludes = ['encrypted_string']
        filtering = {
            'name': ALL
        }

        # TODO: This needs to be made secure when we have a good cert for https
        authentication = IonAuthentication(secure_only=False)
        authorization = DjangoAuthorization()
