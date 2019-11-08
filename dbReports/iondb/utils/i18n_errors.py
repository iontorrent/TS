from django.utils.translation import ugettext_lazy as _
from iondb.utils.validation import _error_prefix


def fatal_internalerror_during_processing(
    fieldOrEntityOrActivityName, include_error_prefix=False, error_prefix=_error_prefix
):
    return _("errors.messages.internalerror.during_processing") % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "name": fieldOrEntityOrActivityName,
    }


def fatal_internalerror_during_processing_of(
    fieldOrEntityOrActivityName,
    item,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return _("errors.messages.internalerror.during_processing_of") % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "name": fieldOrEntityOrActivityName,
        "item": item,
    }


def fatal_internalerror_during_save(
    fieldOrEntityOrActivityName, include_error_prefix=False, error_prefix=_error_prefix
):
    return _("errors.messages.internalerror.during_save") % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "name": fieldOrEntityOrActivityName,
    }


def fatal_internalerror_during_save_of(
    fieldOrEntityOrActivityName,
    item,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return _("errors.messages.internalerror.during_save_of") % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "name": fieldOrEntityOrActivityName,
        "item": item,
    }


def fatal_unexpectederror_during_save(
    fieldOrEntityOrActivityName, include_error_prefix=False, error_prefix=_error_prefix
):
    return _("errors.messages.unexpectederror.during_save") % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "name": fieldOrEntityOrActivityName,
    }


def fatal_unexpectederror_during_save_of(
    fieldOrEntityOrActivityName,
    item,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return _("errors.messages.unexpectederror.during_save_of") % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "name": fieldOrEntityOrActivityName,
        "item": item,
    }


def validationerrors_cannot_save(
    fieldOrEntityOrActivityName, include_error_prefix=False, error_prefix=_error_prefix
):
    return _("errors.messages.validationerrors.cannot_save") % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "name": fieldOrEntityOrActivityName,
    }


def validationerrors_cannot_save_of(
    fieldOrEntityOrActivityName,
    item,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return _("errors.messages.validationerrors.cannot_save_of") % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "name": fieldOrEntityOrActivityName,
        "item": item,
    }


def fatal_unsupported_http_method(
    request_method, include_error_prefix=False, error_prefix=_error_prefix
):
    return _("errors.messages.unsupported.http_method") % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "request_method": request_method,
    }


def fatal_unsupported_http_method_expected(
    request_method,
    expected_method,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return _("errors.messages.unsupported.http_method_expected") % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "request_method": request_method,
        "expected_method": expected_method,
    }
