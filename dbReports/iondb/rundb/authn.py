# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import os
import json

from django.conf import settings
from django.http import HttpResponse, HttpResponseNotFound, HttpResponseRedirect, Http404
from django.contrib.auth.models import AnonymousUser, User

from tastypie.authentication import BasicAuthentication, ApiKeyAuthentication, SessionAuthentication, MultiAuthentication, Authentication
from tastypie.http import *

from iondb.rundb.models import PluginResult

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

# Custom Authorization Class
class IonAuthentication(MultiAuthentication):
    """
        Derived from MultiAuthentication, but with Auth Schemes hardcoded
    """
    def __init__(self, allow_get=None, **kwargs):
        backends = [
            # Basic must be first, so it sets WWW-Authenticate header in 401.
            BasicAuthentication(realm='Torrent Browser'), ## Apache Basic Auth
            SessionAuthentication(),
            PluginApiKeyAuthentication(),
            ApiKeyAuthentication(),
        ]
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

