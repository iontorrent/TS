import os
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from pyftpdlib.authorizers import AuthenticationFailed
from . import _unix

PERMMISSIONS = 'elradfmw'

class FTPAccountAuthorizer(object):
    """Authorizer class by django authentication."""
    personate_user_class = None

    def __init__(self, file_access_user=None):
        if file_access_user:
            personate_user_class = (self.personate_user_class or _unix.UnixPersonateUser)
            self.personate_user = personate_user_class(file_access_user)
        else:
            self.personate_user = None

    def has_user(self, username):
        """return True if exists user."""
        return User.objects.filter(username=username).exists()

    def validate_authentication(self, username, password, handler):
        """authenticate user with password"""

        if not authenticate(username=username, password=password):
            raise AuthenticationFailed("Authentication failed.")

        try:
            return User.objects.get(username=username)
        except User.DoesNotExist:
            raise AuthenticationFailed("Authentication failed.")

    def get_home_dir(self, username):
        """Get the home directory"""
        return '/'

    def get_msg_login(self, username):
        """message for welcome."""
        return 'welcome.'

    def get_msg_quit(self, username):
        """The quit message"""
        return 'good bye.'

    def has_perm(self, username, perm, path=None):
        """check user permission"""
        return perm in PERMMISSIONS

    def get_perms(self, username):
        """return user permissions"""
        return PERMMISSIONS

    def impersonate_user(self, username, password):
        """delegate to personate_user method"""
        if self.personate_user:
            self.personate_user.impersonate_user(username, password)

    def terminate_impersonation(self, username):
        """delegate to terminate_impersonation method"""
        if self.personate_user:
            self.personate_user.terminate_impersonation(username)
