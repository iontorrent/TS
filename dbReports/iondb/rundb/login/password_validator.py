import re
from django.forms import ValidationError
from django.utils.translation import ugettext_lazy as _
from iondb.rundb.models import Rig
from iondb.utils.utils import get_instrument_info

class PasswordValidator:
    validation_methods = ["symbol", "number", "uppercase", "lowercase", "minLength", "blank",
                            "username", "checkInstrumentName", "sameCharacters", "instrumentSupportedSymbol"]
    def __init__(self, password, min_length=10):
        self.min_length = min_length
        self.password = password

    def minLength(self):
        if len(self.password) < self.min_length:
            return ValidationError(
                _("This password must contain at least %(min_length)d characters."),
                code='password_too_short',
                params={'min_length': self.min_length},
            ).messages

    def number(self):
        if not re.findall('\d', self.password):
            return ValidationError(
                _("The password must contain at least 1 digit, 0-9."),
                code='password_no_number',
            ).messages

    def checkInstrumentName(self):
        rigs = Rig.objects.exclude(host_address="")

        if len(rigs) > 0:
            instruments = [get_instrument_info(rig) for rig in rigs]
            for inst in instruments:
                if inst['name'].lower() in self.password.lower() or inst['serial'] in self.password.lower():
                    return ValidationError(
                        _("The password must not contain the instrument name or serial no."),
                        code='password_no_rig_name_serial',
                    ).messages

    def sameCharacters(self):
        regex = "([a-zA-Z0-9.,@#$%^&*()_\-+!;])\\1\\1+"
        p = re.compile(regex)
        if (re.search(p, self.password)):
            return ValidationError(
                _("The password should not contain repeated characters, Ex:aaa,###. Choose different password"),
                code='password_no_sameChars',
            ).messages

    def uppercase(self):
        if not re.findall('[A-Z]', self.password):
            return ValidationError(
                _("The password must contain at least 1 uppercase letter, A-Z."),
                code='password_no_upper',
            ).messages

    def lowercase(self):
        if not re.findall('[a-z]', self.password):
            return ValidationError(
                _("The password must contain at least 1 lowercase letter, a-z."),
                code='password_no_lower',
            ).messages

    def symbol(self):
        if not re.findall('[.,@#$%^&*()_\-+!;]', self.password):
            return ValidationError(
                _("The password must contain at least 1 symbol: .,@#$%^&*()_-+!;"),
                code='password_no_symbol',
            ).messages

    def blank(self):
        if re.search('\s', self.password):
            return ValidationError(
                _("The password must not contain blank space"),
                code='password_no_blank',
            ).messages

    def username(self, username):
        if username.lower() in self.password.lower():
            return ValidationError(
                _("The password must not contain the username"),
                code='password_no_username',
            ).messages

    def instrumentSupportedSymbol(self):
        supported_symbols = ".,@#$%^&*()_-+!;"
        if [e for e in self.password if not e.isalnum() and e not in supported_symbols]:
            return ValidationError(
                _("The password must contain only these symbols supported by the Instruments: .,@#$%^&*()_-+!;"),
                code='password_invalid_symbol',
            ).messages