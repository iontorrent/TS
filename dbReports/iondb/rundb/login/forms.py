# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.models import User as AuthUser
from iondb.rundb import labels
from iondb.rundb.login.password_validator import PasswordValidator


class UserRegistrationForm(
    UserCreationForm
):  # Extends django.contrib.auth.forms.UserCreationForm
    """
    Form for registering a new user account.

    Validates that the requested username is not already in use, and
    requires the password to be entered twice to catch typos.

    """
    password1 = forms.CharField(label=_("Password"),
                                widget=forms.PasswordInput,
                                help_text=_("The password must contain at least a uppercase letter, a digit, special chars(.,@#$%^&*()_-+!;) and minimum 10 characters."))
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
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user

    def clean_password2(self):
        password2 = str(self.cleaned_data.get("password2"))
        if password2:
            passwordValidator = PasswordValidator(password2)
            exec_validation_methods = PasswordValidator.validation_methods
            error_list = {}
            for method in exec_validation_methods:
                if 'username' in method:
                    username = str(self.cleaned_data.get("username"))
                    message = getattr(passwordValidator, method)(username)
                else:
                    message = getattr(passwordValidator, method)()

                if isinstance(message, list):
                    error_list[method] = str(''.join(message))
                elif message is not None:
                    error_list[method] = message
            if error_list:
                raise forms.ValidationError(map(str, error_list.values()))

        return password2

class AuthenticationRememberMeForm(AuthenticationForm):

    """
    Subclass of Django ``AuthenticationForm`` which adds a remember me
    checkbox.
    
    """

    remember_me = forms.BooleanField(
        label=_("Remember Me"), initial=False, required=False
    )