# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from django.conf import settings
from django.contrib.auth.models import User
from tastypie.authentication import BasicAuthentication, ApiKeyAuthentication, SessionAuthentication, MultiAuthentication, Authentication
from tastypie.http import *

from iondb.rundb.models import PluginResult, IonMeshNode


def _extract_system_id_and_key(request):
    """Extracts the system id (if there is one) from the request"""
    meta_auth = request.META.get('HTTP_AUTHORIZATION', '').lower()
    if meta_auth.startswith('system_id '):
        (auth_type, data) = request.META['HTTP_AUTHORIZATION'].split()
        system_id, api_key = data.split(':', 1)
    else:
        system_id = request.GET.get('system_id') or request.POST.get('system_id')
        api_key = request.GET.get('api_key') or request.POST.get('api_key')

    return system_id, api_key


class PluginApiKeyAuthentication(Authentication):

    def _unauthorized(self):
        return HttpUnauthorized()

    def is_authenticated(self, request, **kwargs):
        unauthorized = True

    def extract_credentials(self, request):
        if request.META.get('HTTP_AUTHORIZATION') and request.META['HTTP_AUTHORIZATION'].lower().startswith('pluginkey '):
            (auth_type, data) = request.META['HTTP_AUTHORIZATION'].split()

            if auth_type.lower() != 'pluginkey':
                raise ValueError("Incorrect authorization header.")

            prpk, api_key = data.split(':', 1)
        else:
            prpk = request.GET.get('pluginresult') or request.POST.get('pluginresult')
            api_key = request.GET.get('api_key') or request.POST.get('api_key')

        return prpk, api_key

    def is_authenticated(self, request, **kwargs):
        """
        Finds the pluginresult and checks their API key.

        Should return either ``True`` if allowed, ``False`` if not or an
        ``HttpResponse`` if you need something custom.
        """
        try:
            prpk, api_key = self.extract_credentials(request)
        except ValueError:
            return self._unauthorized()

        if not prpk or not api_key:
            return self._unauthorized()

        try:
            lookup_kwargs = {'id': prpk, 'apikey': api_key}
            pr = PluginResult.objects.get(**lookup_kwargs)
        except (PluginResult.DoesNotExist, PluginResult.MultipleObjectsReturned):
            return self._unauthorized()

        try:
            user = User.objects.get(username='ionuser')
        except (User.DoesNotExist, User.MultipleObjectsReturned):
            user = User.objects.get(username='ionadmin')

        # Allow Active Plugins to act on behalf of ionuser.
        request.user = user
        return True


class IonAuthentication(MultiAuthentication):

    """
        Derived from MultiAuthentication, but with Auth Schemes hardcoded
    """

    # set to true for secure connections only
    secure_only = False

    def __init__(self, allow_get=None, ion_mesh_data_type='', secure_only=False, **kwargs):
        backends = [
            # Basic must be first, so it sets WWW-Authenticate header in 401.
            BasicAuthentication(realm='Torrent Browser'), ## Apache Basic Auth
            SessionAuthentication(),
            PluginApiKeyAuthentication(),
            ApiKeyAuthentication(),
        ]

        self.secure_only = secure_only
        # if this resource is shared as part of the ion mesh network then we need to add an
        # extra authenticaiton method to check for the keys
        if ion_mesh_data_type:
            backends.append(IonMeshAuthentication(data_type=ion_mesh_data_type))

        if allow_get is None:
            allow_get = getattr(settings, 'IONAUTH_ALLOW_REST_GET', False)
        self.allow_get = allow_get
        super(IonAuthentication, self).__init__(*backends, **kwargs)
        # NB: If BasicAuthentication is last, will return WWW-Authorize header and prompt for credentials

    def is_authenticated(self, request, **kwargs):
        """
        Identifies if the user is authenticated to continue or not.

        Should return either ``True`` if allowed, ``False`` if not or an
        ``HttpResponse`` if you need something custom.
        """

        # check the secure only authentication
        if self.secure_only and not request.is_secure():
            return False

        # Allow user with existing session - django authn via session cookie
        if hasattr(request, 'user') and request.user.is_authenticated():
            return True

        # Explicitly disable CSRF for SessionAuthentication POST.
        # This is required unless CSRF view middleware is active
        request._dont_enforce_csrf_checks = True

        unauthorized = super(IonAuthentication, self).is_authenticated(request, **kwargs)

        # Allow GET OPTIONS HEAD without auth
        if self.allow_get and request.method in ('GET', 'OPTIONS', 'HEAD'):
            # Set AnonymousUser?
            return True

        return unauthorized

    def get_identifier(self, request):
        """
        Provides a unique string identifier for the requestor. Delegates to Authn classes
        """
        try:
            return request._authentication_backend.get_identifier(request)
        except AttributeError:
            return 'nouser'


class IonMeshAuthentication(Authentication):
    """This will authenticate based on ion mesh authentication"""

    # a method to check against the api key
    data_type = ''

    def __init__(self, data_type, **kwargs):
        """Constructor: data_type must be 'data', 'plans' or 'monitoring' """
        super(IonMeshAuthentication, self).__init__(**kwargs)
        self.data_type = data_type


    def is_authenticated(self, request, **kwargs):
        """This will attempt to match the request to a known ion mesh node entry"""

        try:
            request.user = User.objects.get(username='ionmesh')
            system_id, api_key = _extract_system_id_and_key(request)
            return IonMeshNode.canAccess(system_id, api_key, self.data_type)
        except:
            return False
