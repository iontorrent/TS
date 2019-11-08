# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.models import User as AuthUser
from iondb.rundb import labels


class UserRegistrationForm(
    UserCreationForm
):  # Extends django.contrib.auth.forms.UserCreationForm
    """
    Form for registering a new user account.

    Validates that the requested username is not already in use, and
    requires the password to be entered twice to catch typos.

    """

    email = forms.EmailField(
        widget=forms.TextInput(attrs={"maxlength": 75}),
        label=labels.User.email.verbose_name,
    )

    def __init__(self, *args, **kwargs):
        super(UserRegistrationForm, self).__init__(*args, **kwargs)
        print(self.fields)
        self.fields["username"].label = labels.User.username.verbose_name
        self.fields["password1"].label = labels.User.password1.verbose_name
        self.fields["password2"].label = labels.User.password2.verbose_name

    def save(self, commit=True):
        user = super(UserRegistrationForm, self).save(commit=False)
        user.set_email(self.cleaned_data["email"])
        if commit:
            user.save()
        return user


class AuthenticationRememberMeForm(AuthenticationForm):

    """
    Subclass of Django ``AuthenticationForm`` which adds a remember me
    checkbox.
    
    """

    remember_me = forms.BooleanField(
        label=_("Remember Me"), initial=False, required=False
    )
