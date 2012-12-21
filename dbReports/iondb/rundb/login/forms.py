# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django import forms
from iondb.rundb import models
from django.contrib.auth.forms import AuthenticationForm
from django.utils.translation import ugettext_lazy as _


class UserRegistrationForm(forms.Form):
    """
    Form for registering a new user account.

    Validates that the requested username is not already in use, and
    requires the password to be entered twice to catch typos.

    """
    username = forms.RegexField(
        regex=r'^[\w.@+-]+$',
        max_length=30,
        widget=forms.TextInput(),
        label="Username",
        error_messages={'invalid':
            "This value may contain only letters, numbers and @/./+/-/_ characters."})

    email = forms.EmailField(widget=forms.TextInput(attrs={'maxlength':75}),
                             label="E-mail")
    password1 = forms.CharField(widget=forms.PasswordInput(render_value=False),
                                label="Password")
    password2 = forms.CharField(widget=forms.PasswordInput(render_value=False),
                                label="Password (again)")

    def clean_username(self):
        """
        Validate that the username is alphanumeric and is not already
        in use.
        """
        existing = models.User.objects.filter(username__iexact=self.cleaned_data['username'])
        if existing.exists():
            raise forms.ValidationError("A user with that username already exists.")
        else:
            return self.cleaned_data['username']

    def clean(self):
        """
        Verify that the values entered into the two password fields
        match. Note that an error here will end up in
        ``non_field_errors()`` because it doesn't apply to a single
        field.
        """
        if 'password1' in self.cleaned_data and 'password2' in self.cleaned_data:
            if self.cleaned_data['password1'] != self.cleaned_data['password2']:
                raise forms.ValidationError("The two password fields didn't match.")
        return self.cleaned_data


class AuthenticationRememberMeForm(AuthenticationForm):

    """
    Subclass of Django ``AuthenticationForm`` which adds a remember me
    checkbox.
    
    """
    
    remember_me = forms.BooleanField(label=_('Remember Me'), initial=False,
        required=False)