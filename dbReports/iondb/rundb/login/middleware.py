# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.contrib.auth.models import User
from django.http import HttpResponse
from django.contrib.auth import authenticate, login

import logging
log = logging.getLogger(__name__)

class BasicAuthMiddleware:
    """
    Simple HTTP-Basic auth for testing webservices

    Used for optional HTTP Auth - validated in django not at apache layer.
    Use in conjunction with session auth to provide alternatives for users.

    """
    def process_request(self, request):
        auth_header = request.META.get('HTTP_AUTHORIZATION', None)
        if auth_header is None:
            # Do nothing, fall through to other methods
            return None

        auth = request.META['HTTP_AUTHORIZATION'].split()
        if len(auth) == 2:
            # NOTE: We are only support basic authentication for now.
            if auth[0].lower() == "basic":
                import base64
                username, password = base64.b64decode(auth[1]).split(':')

                user = authenticate(username=username, password=password)
                if user is not None:
                    if user.is_active:
                        login(request, user)
                    else:
                        log.debug("Failed login attempt for user '%s'. INACTIVE USER", user)
            else:
                log.debug("Attempt to auth with '%s' auth: NOT SUPPORTED", auth[0])
        else:
            log.debug("Unrecognized HTTP_AUTHORIZATION header received: '%s'", request.META['HTTP_AUTHORIZATION'])

        return None
