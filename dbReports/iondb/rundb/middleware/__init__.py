# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import logging

from django.conf import settings
from django.contrib import auth
from django.contrib.auth import load_backend
from django.contrib.auth.backends import RemoteUserBackend
from django.contrib.auth.middleware import RemoteUserMiddleware
from django.core.exceptions import ImproperlyConfigured
from django.shortcuts import redirect

logger = logging.getLogger(__name__)

public_url_names = []
if hasattr(settings, "PUBLIC_URL_NAMES"):
    public_url_names += [url for url in settings.PUBLIC_URL_NAMES]


class ChangeRequestMethodMiddleware(object):
    def process_request(self, request):
        if request.method == "PROPPATCH":
            request.method = "PATCH"
        return


class LocalhostAuthMiddleware(object):

    """ If request originates from localhost, autologin as localuser
        Complements Allow records in apache basic auth """

    header = "REMOTE_ADDR"
    localuser = "ionadmin"

    def process_request(self, request):
        if not hasattr(request, "user"):
            raise ImproperlyConfigured(
                "The Django remote user auth middleware requires the"
                " authentication middleware to be installed.  Edit your"
                " MIDDLEWARE_CLASSES setting to insert"
                " 'django.contrib.auth.middleware.AuthenticationMiddleware'"
                " before the RemoteUserMiddleware class."
            )

        if self.header in request.META:
            remote = request.META[self.header]
        else:
            # No usable HEADER
            return

        if remote not in ["127.0.0.1", "127.0.1.1", "::1"]:
            return

        # Keep existing user, allow it to pass through REMOTE_USER
        # otherwise RemoteUserMiddleware logs the user out again!
        if request.user.is_authenticated():
            try:
                stored_backend = load_backend(
                    request.session.get(auth.BACKEND_SESSION_KEY, "")
                )
                if isinstance(stored_backend, RemoteUserBackend):
                    # Fake it for remote user
                    request.META.setdefault(
                        RemoteUserMiddleware.header, request.user.username
                    )
            except ImproperlyConfigured as e:
                # backend failed to load
                auth.logout(request)
            # Otherwise keep currently logged in user - no need to override it
            return

        # We are seeing this user for the first time in this session, attempt
        # to authenticate the user. (via RemoteUser backend)
        user = auth.authenticate(remote_user=self.localuser)
        if user:
            # User is valid.  Set request.user and persist user in the session
            # by logging the user in.
            request.user = user
            request.META.setdefault(RemoteUserMiddleware.header, request.user.username)
            auth.login(request, user)


"""Delete sessionid and csrftoken cookies on logout, for better compatibility with upstream caches."""


class DeleteSessionOnLogoutMiddleware(object):
    def process_response(self, request, response):
        if getattr(request, "_delete_session", False):
            logger.info(
                "DeleteSessionOnLogoutMiddleware.process_response: Deleting Cookies"
            )
            response.delete_cookie(
                settings.CSRF_COOKIE_NAME, domain=settings.CSRF_COOKIE_DOMAIN
            )
            response.delete_cookie(
                settings.SESSION_COOKIE_NAME,
                settings.SESSION_COOKIE_PATH,
                settings.SESSION_COOKIE_DOMAIN,
            )
        return response

    def process_view(self, request, view_func, view_args, view_kwargs):
        try:
            view_name = ".".join((view_func.__module__, view_func.__name__))
            # flag for deletion if this is a logout view
            request._delete_session = view_name in (
                "django.contrib.admin.sites.logout",
                "django.contrib.auth.views.logout",
                "iondb.rundb.login.views.logout_view",
            )
            if request._delete_session:
                logger.info(
                    "DeleteSessionOnLogoutMiddleware.process_view: request._delete_session=%s"
                    % request._delete_session
                )
        except AttributeError:
            pass  # if view_func doesn't have __module__ or __name__ attrs
        return None


class LoginRequiredMiddleware(object):
    """
       Middleware that requires a user to be authenticated to view any page other
       than LOGIN_URL. Exemptions to this requirement can optionally be specified
       in settings via a list of regular expressions in PUBLIC_URL_NAMES.
       For REST APIs : let the api auth methods handle it (/iondb/rundb/authn.py)
    """

    def process_view(self, request, view_func, view_args, view_kwargs):
        try:
            if not hasattr(request, "user"):
                raise ImproperlyConfigured(
                    "The Login Required middleware needs to be after AuthenticationMiddleware."
                )

            current_url = request.resolver_match.url_name
            if current_url == "logout":
                return redirect(settings.LOGIN_URL)
            if request.user.is_authenticated() or current_url in public_url_names:
                return None
            elif request.resolver_match.kwargs.get("api_name") == "v1":
                return None
            else:
                return redirect(settings.LOGIN_URL + "?next=" + request.path)

        except Exception as Err:
            logger.info("Exception :LoginRequiredMiddleware.process_view %s", Err)

        return None
