import locale
import logging
import re

from django.conf import settings
from django.utils import translation
from django.utils.datastructures import SortedDict

logger = logging.getLogger(__name__)

_accepted = {}  # becomes a global


class DefaultSDKLanguageMiddleware(object):
    """
    Introduce a Middleware to default the SDK (Tastypie), using a pattern matching,
    to activate a default SDK specific language code, see settings.SDK_LANGUAGE_CODE.

    Adapted from 'django.middleware.locale.LocaleMiddleware'.


    A simple middleware that parses a request and decides what translation object to install in the current
    thread context. This allows SDK to be dynamically translated to the language the client desires (if the language
    is available, of course).
    """

    def __init__(self):
        self._default_sdk_language = (
            settings.SDK_LANGUAGE_CODE
            if settings.SDK_LANGUAGE_CODE
            else settings.LANGUAGE_CODE
        )
        self.PUBLIC_URL_PATTERNS = [r"^/rundb/api/.*$"]
        self.PUBLIC_URL_PATTERNS = [re.compile(exp) for exp in self.PUBLIC_URL_PATTERNS]

    def _get_language_from_request(self, request, check_path=False):
        """
        Adapted from django/utils/translation/trans_real.py

        :return: the language from Accept-Language header if supported, otherwise returns the self._default_sdk_language

        ----

        Analyzes the request to find what language the user wants the system to
        show. Only languages listed in settings.LANGUAGES are taken into account.
        If the user requests a sublanguage where we have a main language, we send
        out the main language.

        If check_path is True, the URL path prefix will be checked for a language
        code, otherwise this is skipped for backwards compatibility.
        """
        # Format of Accept-Language header values. From RFC 2616, section 14.4 and 3.9
        # and RFC 3066, section 2.1
        accept_language_re = re.compile(
            r"""
                ([A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*|\*)      # "en", "en-au", "x-y-z", "es-419", "*"
                (?:\s*;\s*q=(0(?:\.\d{,3})?|1(?:.0{,3})?))?   # Optional "q=1.00", "q=0.8"
                (?:\s*,\s*|$)                                 # Multiple accepts per header.
                """,
            re.VERBOSE,
        )

        def to_locale(language, to_lower=False):
            """
            Turns a language name (en-us) into a locale name (en_US). If 'to_lower' is
            True, the last component is lower-cased (en_us).
            """
            p = language.find("-")
            if p >= 0:
                if to_lower:
                    return language[:p].lower() + "_" + language[p + 1 :].lower()
                else:
                    # Get correct locale for sr-latn
                    if len(language[p + 1 :]) > 2:
                        return (
                            language[:p].lower()
                            + "_"
                            + language[p + 1].upper()
                            + language[p + 2 :].lower()
                        )
                    return language[:p].lower() + "_" + language[p + 1 :].upper()
            else:
                return language.lower()

        def parse_accept_lang_header(lang_string):
            """
            Parses the lang_string, which is the body of an HTTP Accept-Language
            header, and returns a list of (lang, q-value), ordered by 'q' values.

            Any format errors in lang_string results in an empty list being returned.
            """
            result = []
            pieces = accept_language_re.split(lang_string)
            if pieces[-1]:
                return []
            for i in range(0, len(pieces) - 1, 3):
                first, lang, priority = pieces[i : i + 3]
                if first:
                    return []
                if priority:
                    priority = float(priority)
                if not priority:  # if priority is 0.0 at this point make it 1.0
                    priority = 1.0
                result.append((lang, priority))
            result.sort(key=lambda k: k[1], reverse=True)
            return result

        def get_supported_language_variant(lang_code, supported=None, strict=False):
            """
            Returns the language-code that's listed in supported languages, possibly
            selecting a more generic variant. Raises LookupError if nothing found.

            If `strict` is False (the default), the function will look for an alternative
            country-specific variant when the currently checked is not found.
            """
            if supported is None:
                from django.conf import settings

                supported = SortedDict(settings.LANGUAGES)
            if lang_code:
                # if fr-CA is not supported, try fr-ca; if that fails, fallback to fr.
                generic_lang_code = lang_code.split("-")[0]
                variants = (
                    lang_code,
                    lang_code.lower(),
                    generic_lang_code,
                    generic_lang_code.lower(),
                )
                for code in variants:
                    if code in supported and translation.check_for_language(code):
                        return code
                if not strict:
                    # if fr-fr is not supported, try fr-ca.
                    for supported_code in supported:
                        if supported_code.startswith(
                            (generic_lang_code + "-", generic_lang_code.lower() + "-")
                        ):
                            return supported_code
            raise LookupError(lang_code)

        global _accepted
        from django.conf import settings

        supported = SortedDict(settings.LANGUAGES)

        # Uncomment and edit if using path-based language support
        # if check_path:
        #     lang_code = get_language_from_path(request.path_info, supported)
        #     if lang_code is not None:
        #         return lang_code

        # Uncomment and edit if using Sessions to track language
        # if hasattr(request, 'session'):
        #     lang_code = request.session.get('django_language', None)
        #     if lang_code in supported and lang_code is not None and check_for_language(lang_code):
        #         return lang_code

        # Uncomment and edit if using Cookies to track language
        # lang_code = request.COOKIES.get(settings.LANGUAGE_COOKIE_NAME)
        #
        # try:
        #     return get_supported_language_variant(lang_code, supported)
        # except LookupError:
        #     pass

        accept = request.META.get("HTTP_ACCEPT_LANGUAGE", "")
        for accept_lang, unused in parse_accept_lang_header(accept):
            if accept_lang == "*":
                break

            # 'normalized' is the root name of the locale in POSIX format (which is
            # the format used for the directories holding the MO files).
            normalized = locale.locale_alias.get(to_locale(accept_lang, True))
            if not normalized:
                continue
            # Remove the default encoding from locale_alias.
            normalized = normalized.split(".")[0]

            if normalized in _accepted:
                # We've seen this locale before and have an MO file for it, so no
                # need to check again.
                return _accepted[normalized]

            try:
                accept_lang = get_supported_language_variant(accept_lang, supported)
            except LookupError:
                continue
            else:
                _accepted[normalized] = accept_lang
                return accept_lang

        try:
            return get_supported_language_variant(self._default_sdk_language, supported)
        except LookupError:
            return self._default_sdk_language

    def _is_url_whitelist(self, request):
        path = request.path
        if any(pattern.match(path) for pattern in self.PUBLIC_URL_PATTERNS):
            return True
        return False

    def process_request(self, request):
        if self._is_url_whitelist(request):
            language = self._get_language_from_request(
                request, check_path=False
            )  # will return the language from the request if provided otherwise returns the self._default_sdk_language
            translation.activate(language)
            request.LANGUAGE_CODE = translation.get_language()

    def process_response(self, request, response):
        if self._is_url_whitelist(request):
            language = translation.get_language()
            if "Content-Language" not in response:
                response["Content-Language"] = language
            translation.deactivate()
        return response

    def process_exception(self, request, exception):
        if self._is_url_whitelist(request):
            language = translation.get_language()
            translation.deactivate()
