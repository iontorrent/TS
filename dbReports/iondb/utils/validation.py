# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import re
from collections import Iterable, namedtuple

from django.utils.encoding import force_text

from iondb.utils import verify_types as types
from django.utils.translation import ugettext as _
from django.utils.translation import ugettext_lazy

_error_prefix = ugettext_lazy("validation.messages.error_prefix")
_warning_prefix = ugettext_lazy("validation.messages.warning_prefix")


class SeparatedValuesBuilder(object):
    def build(self, value, separator=", "):
        """

        :param value: the object (string, list, dict, tuple, Django type...
        :param separator: the value to join the value
        :return: unicode of value OR a separated unicode string of elements of value
        """
        if isinstance(value, basestring):
            return unicode(value)
        elif isinstance(value, Iterable):
            return unicode(separator.join(value))
        else:
            return unicode(value)


_separated_values_builder = SeparatedValuesBuilder()


class Rules(object):
    EMPTY = "empty"
    EMPTY_RELATED = "empty_related"
    EMPTY_RELATED_VALUE = "empty_related_value"
    REQUIRED = "required"  # input required
    REQUIRED_VALUE_NOT_POLYMORPHIC_TYPE_VALUE = (
        "required_value_not_polymorphic_type_value"
    )  # input required
    REQUIRED_AT_LEAST_ONE_POLYMORPHIC_TYPE_VALUE = (
        "required_at_least_one_polymorphic_type_value"
    )  # input required
    REQUIRED_AT_LEAST_ONE = "required_at_least_one"
    REQUIRED_AT_LEAST_ONE_CHILD = "required_at_least_one_child"
    REQUIRED_RELATED = "required_related"  # input required given related
    REQUIRED_RELATED_VALUE = (
        "required_related_value"
    )  # input required given related value is
    MISSING = "missing"  # missing
    INT = "int"  # integer
    UINT = "uint"  # integer > 0
    UINT_N_ZERO = "uint_n_zero"  # integer >= 0
    ALPHANUM = "alphanum"  # letters and numbers
    NAME = "name"  # letters, numbers, space, and .-_
    SLUG = "slug"  # letters, numbers, underscores or hyphens
    LEAD_CHAR = "lead_char"  # lead character must be letter or number
    MIN_LENGTH = "min_length"  # min length
    MAX_LENGTH = "max_length"  # max length
    NUCLEOTIDE = "nucleotide"  # ATCG
    KEYWORDS = "keywords"  # compare value to a fixed list of supported values
    NOT_FOUND = "not_found"  # input not recognized
    ENTITY_DATA = "entity_data"
    ENTITY_FIELD_UNIQUE = "entity_field_unique"
    ENTITY_FIELD_UNIQUE_VALUE = "entity_field_unique_value"
    INVALID_VALUE = "invalid_value"
    INVALID_RELATED = "invalid_related"
    INVALID_VALUE_RELATED = "invalid_value_related"
    INVALID_VALUE_RELATED_VALUE = "invalid_value_related_value"
    INVALID_CHOICE = "invalid_choice"
    INVALID_CHOICE_RELATED = "invalid_choice_related"
    INVALID_CHOICE_RELATED_CHOICE = "invalid_choice_related_choice"
    MAX_VALUE = "max_value"
    MIN_VALUE = "min_value"
    NO_DATA = "no_data"
    NO_DATA_FROM = "no_data_from"
    NOT_ACTIVE = "not_active"
    RANGE = "range"
    ROW_ERRORS = "row_errors"
    NOT_CONFIGURED = "not_configured"
    NOT_REACHABLE = "not_reachable"
    NOT_REACHABLE_NOT_CONFIGURED = "not_reachable_not_configured"
    PROVIDED = "provided"
    SCHEMA_MISSING_ATTRIBUTE = "schema_missing_attribute"
    FILE_DOES_NOT_EXIST = "file_does_not_exist"
    FILE_INVALID_EXTENSION = "file_invalid_extension"
    FILE_MISSING_EXTENSION = "file_missing_extension"
    FILE_EMPTY = "file_empty"
    INVALID_DATE_FORMAT = "invalid_date_format"
    INVALID_RECEIPT_DATE = "invalid_receipt_date"

    @classmethod
    def get_error(cls, rule):
        msg = {
            cls.EMPTY: _("validators.messages.empty"),  # ' is empty'
            cls.EMPTY_RELATED: _(
                "validators.messages.empty_related"
            ),  # '%(fieldName)s is empty for %(relatedName)s'
            cls.EMPTY_RELATED_VALUE: _(
                "validators.messages.empty_related_value"
            ),  # '%(fieldName)s is empty for %(relatedName)s with a value of %(relatedValue)s'
            cls.REQUIRED: _("validators.messages.required"),  # ' is required'
            cls.REQUIRED_VALUE_NOT_POLYMORPHIC_TYPE_VALUE: _(
                "validators.messages.required_value_not_polymorphic_type_value"
            ),  # ' is required'
            cls.REQUIRED_AT_LEAST_ONE_POLYMORPHIC_TYPE_VALUE: _(
                "validators.messages.required_at_least_one_polymorphic_type_value"
            ),  # ' is required'
            cls.REQUIRED_AT_LEAST_ONE: _(
                "validators.messages.required_at_least_one"
            ),  # 'At least one %(fieldName)s is required. '
            cls.REQUIRED_AT_LEAST_ONE_CHILD: _(
                "validators.messages.required_at_least_one_child"
            ),  # '%(parentName)s requires at least one %(childName)s. '. '
            cls.REQUIRED_RELATED: _(
                "validators.messages.required_related"
            ),  # '%(fieldName)s is required for this %(relatedName)s. '
            cls.REQUIRED_RELATED_VALUE: _(
                "validators.messages.required_related_value"
            ),  # '%(fieldName)s is required for this %(relatedName)s with value of %(relatedValue)s"'
            cls.MISSING: _("validators.messages.missing"),  # ' is missing'
            cls.INT: _("validators.messages.int"),  # ' must be an integer',
            cls.UINT: _("validators.messages.uint"),  # ' must be a positive integer',
            cls.UINT_N_ZERO: _(
                "validators.messages.uint_n_zero"
            ),  # ' must be a non-negative integer',
            cls.ALPHANUM: _(
                "validators.messages.alphanum"
            ),  # ' should contain only numbers and letters (without special characters)',
            cls.NAME: _(
                "validators.messages.name"
            ),  # ' should contain only letters, numbers, spaces, and the following: . - _ ',
            cls.SLUG: _(
                "validators.messages.slug"
            ),  # ' should contain only letters, numbers, underscores or hyphens',
            cls.LEAD_CHAR: _(
                "validators.messages.lead_char"
            ),  # ' should start with letter or number',
            cls.MIN_LENGTH: _(
                "validators.messages.min_length"
            ),  # ' should be at least %s characters long',
            cls.MAX_LENGTH: _(
                "validators.messages.max_length"
            ),  # ' should not be longer than %s characters',
            cls.NUCLEOTIDE: _(
                "validators.messages.nucleotide"
            ),  # ' valid values are TACG',
            cls.KEYWORDS: _("validators.messages.keywords"),  # ' valid values are ',
            cls.NOT_FOUND: _("validators.messages.not_found"),  # " not found for %s ",
            cls.ENTITY_DATA: _(
                "validators.messages.entity_data"
            ),  # " No %(entity)s data to validate. ",
            cls.ENTITY_FIELD_UNIQUE: _(
                "validators.messages.entity_field_unique"
            ),  # " %(model_name)s with this %(field_label)s already exists.",
            cls.ENTITY_FIELD_UNIQUE_VALUE: _(
                "validators.messages.entity_field_unique_value"
            ),  # " %(model_name)s with this %(field_label)s value of %(field_value)s already exists.",
            cls.INVALID_VALUE: _(
                "validators.messages.invalid_value"
            ),  # '%(fieldName)s value of %(fieldValue)s is not valid.'
            cls.INVALID_RELATED: _(
                "validators.messages.invalid_related"
            ),  # '%(fieldName)s is not supported for this %(relatedName)s'
            cls.INVALID_VALUE_RELATED: _(
                "validators.messages.invalid_value_related"
            ),  # '%(fieldName)s value of %(fieldValue)s is not supported for this %(relatedName)s'
            cls.INVALID_VALUE_RELATED_VALUE: _(
                "validators.messages.invalid_value_related_value"
            ),  # '%(fieldName)s value of %(fieldValue)s is not supported for this %(relatedName)s'
            cls.INVALID_CHOICE: _(
                "validators.messages.invalid_choice"
            ),  # '%(fieldName)s value is not valid. %(value)s is not one of the available choices %(choices)s.'
            cls.INVALID_CHOICE_RELATED: _(
                "validators.messages.invalid_choice_related"
            ),  # '%(entity)s is not active.'
            cls.INVALID_CHOICE_RELATED_CHOICE: _(
                "validators.messages.invalid_choice_related_choice"
            ),  # '%(entity)s is not active.'
            cls.MAX_VALUE: _(
                "validators.messages.max_value"
            ),  # '%(fieldName)s must be less than or equal to %(limit_value)s.'
            cls.MIN_VALUE: _("validators.messages.min_value"),
            # '%(fieldName)s must be greater than or equal to %(limit_value)s.'
            cls.NO_DATA: _(
                "validators.messages.no_data"
            ),  # ' No %(fieldName)s data to validate.'
            cls.NO_DATA: _(
                "validators.messages.no_data_from"
            ),  # 'No %(name)s data found in %(item)s.'
            cls.NOT_ACTIVE: _(
                "validators.messages.not_active"
            ),  # '%(entity)s is not active.'
            cls.RANGE: _("validators.messages.range"),  # '%(entity)s is not active.'
            cls.ROW_ERRORS: _(
                "validators.messages.row_errors"
            ),  # 'Error in row %(rowNumber)s: %(rowErrors)s'
            cls.NOT_CONFIGURED: _(
                "validators.messages.not_configured"
            ),  # 'Error in row %(rowNumber)s: %(rowErrors)s'
            cls.NOT_REACHABLE: _(
                "validators.messages.not_reachable"
            ),  # 'Error in row %(rowNumber)s: %(rowErrors)s'
            cls.NOT_REACHABLE_NOT_CONFIGURED: _(
                "validators.messages.not_reachable_not_configured"
            ),  # 'Error in row %(rowNumber)s: %(rowErrors)s'
            cls.PROVIDED: _(
                "validators.messages.provided"
            ),  # '%(fieldName)s data is provided. '
            cls.SCHEMA_MISSING_ATTRIBUTE: _(
                "validators.messages.schema_missing_attribute"
            ),  # 'Schema Error: %(entityName)s is missing %(fieldName)s attribute'
            cls.FILE_DOES_NOT_EXIST: _(
                "validators.messages.file_does_not_exist"
            ),  # 'File %(file)s does not exist. '
            cls.FILE_INVALID_EXTENSION: _(
                "validators.messages.file_invalid_extension"
            ),  # '%(choice)s is not one of the supported file types %(choices)s. '
            cls.FILE_MISSING_EXTENSION: _(
                "validators.messages.file_missing_extension"
            ),  # 'The file extension is missing from file name. Unable to identify type of file. '
            cls.FILE_EMPTY: _(
                "validators.messages.file_empty"
            ),  # '%(label)s %(file)s is empty, zero size and has no contents. '
            cls.INVALID_DATE_FORMAT: _(
                "validators.messages.invalid_date_format"
            ),  # '%(date)s is not in valid format.
            cls.INVALID_RECEIPT_DATE: _(
                "validators.messages.invalid_receipt_date"
            ),  # '%(sample receipt date)s is not valid.
            "unknown": _("validators.messages.unknown"),
            # ' failed to validate. Validator function "%s" is not defined'
        }
        if rule in msg:
            return msg[rule]
        else:
            return msg["unknown"] % {"rule": rule}

    @classmethod
    def validate(cls, rule, value, arg=""):
        if rule == cls.REQUIRED:
            if isinstance(value, str):
                value = value.strip()
            return True if value else False
        elif rule == cls.INT:
            return types.RepresentsInt(value)
        elif rule == cls.UINT:
            return types.RepresentsUnsignedInt(value)
        elif rule == cls.UINT_N_ZERO:
            return types.RepresentsUnsignedIntOrZero(value)
        elif rule == cls.ALPHANUM:
            return bool(re.match("^$|[\w]+$", value))
        elif rule == cls.NAME:
            return bool(re.match("^$|[a-zA-Z0-9\\-_. ]+$", value))
        elif rule == cls.SLUG:
            return bool(re.match("^$|[a-zA-Z0-9\\-_]+$", value))
        elif rule == cls.LEAD_CHAR:
            return bool(re.match("^$|[a-zA-Z0-9]$", value.strip()[0]))
        elif rule == cls.MIN_LENGTH:
            return len(value.strip()) >= arg
        elif rule == cls.MAX_LENGTH:
            return len(value.strip()) <= arg
        elif rule == cls.NUC:
            return bool(re.match("^$[ATCG]+$", value))

        return False


""" Validation helper functions """


def is_valid_chars(value):
    # only letters, numbers, spaces, dashes, underscores and dots
    return Rules.validate(Rules.NAME, value)


def invalid_chars_error(
    fieldName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.NAME) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def is_valid_leading_chars(value):
    # leading chars must be letters or numbers
    if value:
        return Rules.validate(Rules.LEAD_CHAR, value)
    return True


def invalid_leading_chars(
    fieldName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.LEAD_CHAR) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def is_valid_length(value, maxLength):
    # value length must be within the maximum allowed
    if value:
        return Rules.validate(Rules.MAX_LENGTH, value, maxLength)
    return True


def invalid_length_error(
    fieldName,
    maxLength,
    currentLength,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return Rules.get_error(Rules.MAX_LENGTH) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "max": maxLength,
        "current": len(currentLength),
    }


def is_valid_minlength(value, minLength):
    # value length must be within the maximum allowed
    if value:
        return Rules.validate(Rules.MIN_LENGTH, value, minLength)
    return True


def invalid_minlegth_error(
    fieldName,
    minLength,
    currentLength,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return Rules.get_error(Rules.MIN_LENGTH) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "min": minLength,
        "current": len(currentLength),
    }


def invalid_keyword_error(
    fieldName,
    valid_keyword_list,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return Rules.get_error(Rules.KEYWORDS) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "keywords": ", ".join(valid_keyword_list),
    }


def invalid_not_found_error(
    fieldName,
    valueOrValues,
    valueOrValues_separator=", ",
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    valueOrValues_str = _separated_values_builder.build(
        valueOrValues, valueOrValues_separator
    )
    return Rules.get_error(Rules.NOT_FOUND) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "value": valueOrValues_str,
    }


def is_valid_int(value):
    # must be  integer
    return Rules.validate(Rules.INT, value)


def invalid_int(fieldName, include_error_prefix=False, error_prefix=_error_prefix):
    return Rules.get_error(Rules.INT) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def is_valid_uint(value):
    # must be non-negative integer
    return Rules.validate(Rules.UINT, value)


def invalid_uint(fieldName, include_error_prefix=False, error_prefix=_error_prefix):
    return Rules.get_error(Rules.UINT) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def is_valid_uint_n_zero(value):
    # must be non-negative integer, zero included
    return Rules.validate(Rules.UINT_N_ZERO, value)


def invalid_uint_n_zero(
    fieldName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.UINT_N_ZERO) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def invalid_alphanum(
    fieldName, fieldValue, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.ALPHANUM) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "fieldValue": fieldValue,
    }


def has_value(value):
    # value is required
    return Rules.validate(Rules.REQUIRED, value)


def invalid_empty(fieldName, include_error_prefix=False, error_prefix=_error_prefix):
    return Rules.get_error(Rules.EMPTY) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def invalid_empty_related(
    fieldName, relatedName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.EMPTY_RELATED) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "relatedName": relatedName,
    }


def invalid_empty_related_value(
    fieldName,
    relatedName,
    relatedValue,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return Rules.get_error(Rules.EMPTY_RELATED_VALUE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "relatedName": relatedName,
        "relatedValue": relatedValue,
    }


def required_error(fieldName, include_error_prefix=False, error_prefix=_error_prefix):
    return Rules.get_error(Rules.REQUIRED) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def invalid_required_value_not_polymorphic_type_value(
    entityName,
    entityFieldName,
    notEntityName,
    notEntityFieldName,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return Rules.get_error(Rules.REQUIRED_VALUE_NOT_POLYMORPHIC_TYPE_VALUE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "entityName": entityName,
        "entityFieldName": entityFieldName,
        "notEntityName": notEntityName,
        "notEntityFieldName": notEntityFieldName,
    }


Entity_EntityFieldName = namedtuple(
    "Entity_EntityFieldName", ["entityName", "entityFieldName"]
)


def invalid_required_at_least_one_polymorphic_type_value(
    Entity_EntityFieldName1,
    Entity_EntityFieldName2,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return Rules.get_error(Rules.REQUIRED_AT_LEAST_ONE_POLYMORPHIC_TYPE_VALUE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "Entity_EntityFieldName1": "%s %s"
        % (
            force_text(Entity_EntityFieldName1.entityName),
            force_text(Entity_EntityFieldName1.entityFieldName),
        ),
        "Entity_EntityFieldName2": "%s %s"
        % (
            force_text(Entity_EntityFieldName2.entityName),
            force_text(Entity_EntityFieldName2.entityFieldName),
        ),
    }


def invalid_required_at_least_one(
    fieldName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.REQUIRED_AT_LEAST_ONE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def invalid_required_at_least_one_child(
    parentName, childName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.REQUIRED_AT_LEAST_ONE_CHILD) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "parentName": parentName,
        "childName": childName,
    }


def invalid_required_related(
    fieldName, relatedName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.REQUIRED_RELATED) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "relatedName": relatedName,
    }


def invalid_required_related_value(
    fieldName,
    relatedName,
    relatedValue,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    _related_value = _separated_values_builder.build(relatedValue, ", ")
    return Rules.get_error(Rules.REQUIRED_RELATED_VALUE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "relatedName": relatedName,
        "relatedValue": _related_value,
    }


def is_valid_keyword(value, valid_keyword_list):
    return value.lower() in (key.lower() for key in valid_keyword_list)


def missing_error(fieldName, include_error_prefix=False, error_prefix=_error_prefix):
    return Rules.get_error(Rules.MISSING) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def valid_entity_data(
    entity, entityName, include_error_prefix=False, error_prefix=_error_prefix
):
    if not entity:  # if falsey
        return Rules.get_error(Rules.ENTITY_DATA) % {
            "errorPrefix": error_prefix if include_error_prefix else "",
            "entity": entityName,
        }
    return None


def invalid_entity_field_unique(
    entityName, fieldName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.ENTITY_FIELD_UNIQUE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "model_name": entityName,
        "field_label": fieldName,
    }


def invalid_entity_field_unique_value(
    entityName,
    fieldName,
    fieldValue,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return Rules.get_error(Rules.ENTITY_FIELD_UNIQUE_VALUE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "model_name": entityName,
        "field_label": fieldName,
        "field_value": fieldValue,
    }


def format(
    lazy_message,
    message_params=None,
    include_error_prefix=False,
    error_prefix=_error_prefix,
    include_warning_prefix=False,
    warning_prefix=_warning_prefix,
):
    message_params = {} if message_params is None else message_params
    message_params["errorPrefix"] = (
        error_prefix if include_error_prefix and not include_warning_prefix else ""
    )
    message_params["warningPrefix"] = (
        warning_prefix if include_warning_prefix and not include_error_prefix else ""
    )
    return lazy_message % message_params


def is_valid_choice(value, valid_choices, insensitive=True):
    if insensitive:
        return value.lower() in (key.lower() for key in valid_choices)
    else:
        return value in (key for key in valid_choices)


def invalid_choice(
    fieldName, choice, choices, include_error_prefix=False, error_prefix=_error_prefix
):
    choices_str = _separated_values_builder.build(choices, ", ")
    return Rules.get_error(Rules.INVALID_CHOICE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "choice": choice,
        "choices": choices_str,
    }


def invalid_choice_related(
    fieldName,
    choice,
    choices,
    relatedName,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    choices_str = _separated_values_builder.build(choices, ", ")
    return Rules.get_error(Rules.INVALID_CHOICE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "choice": choice,
        "choices": choices_str,
        "relatedName": relatedName,
    }


def invalid_choice_related_choice(
    fieldName,
    choice,
    choices,
    relatedName,
    relatedChoice,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    choices_str = _separated_values_builder.build(choices, ", ")
    return Rules.get_error(Rules.INVALID_CHOICE_RELATED_CHOICE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "choice": choice,
        "choices": choices_str,
        "relatedName": relatedName,
        "relatedChoice": relatedChoice,
    }


def invalid_min_value(
    fieldName, limit_value, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.MIN_VALUE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "limit_value": limit_value,
    }


def invalid_max_value(
    fieldName, limit_value, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.MAX_VALUE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "limit_value": limit_value,
    }


def invalid_no_data(
    entityDataName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.NO_DATA) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "entity": entityDataName,
    }


def invalid_no_data_from(
    fieldOrEntityOrActivityName,
    item,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return Rules.get_error(Rules.NO_DATA_FROM) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "name": fieldOrEntityOrActivityName,
        "item": item,
    }


def invalid_invalid_related(
    fieldName, relatedName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.INVALID_RELATED) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "relatedName": relatedName,
    }


def invalid_invalid_value(
    fieldName, fieldValue, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.INVALID_VALUE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "fieldValue": fieldValue,
    }


def invalid_invalid_value_related(
    fieldName,
    fieldValue,
    relatedName,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return Rules.get_error(Rules.INVALID_VALUE_RELATED) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "fieldValue": fieldValue,
        "relatedName": relatedName,
    }


def invalid_invalid_value_related_value(
    fieldName,
    fieldValue,
    relatedName,
    relatedValue,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    _related_value = _separated_values_builder.build(relatedValue, ", ")
    return Rules.get_error(Rules.INVALID_VALUE_RELATED_VALUE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "fieldValue": fieldValue,
        "relatedName": relatedName,
        "relatedValue": _related_value,
    }


def invalid_not_active(
    entityName, entityValue, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.NOT_ACTIVE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "entity": entityName,
        "entityValue": entityValue,
    }


def invalid_range(
    fieldName,
    min_value,
    max_value,
    include_error_prefix=False,
    error_prefix=_error_prefix,
):
    return Rules.get_error(Rules.RANGE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "min_value": min_value,
        "max_value": max_value,
    }


def invalid_nucleotide(
    fieldName, fieldValue, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.NUCLEOTIDE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "fieldValue": fieldValue,
    }


def invalid_date_format(
    fieldName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.INVALID_DATE_FORMAT) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def invalid_receipt_date(
    fieldName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.INVALID_RECEIPT_DATE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def row_errors(rowNumber, rowErrors, rowErrors_separator=" ", ending="\n"):
    rowErrors_str = _separated_values_builder.build(rowErrors, rowErrors_separator)
    result = Rules.get_error(Rules.ROW_ERRORS) % {
        "rowNumber": rowNumber,
        "rowErrors": rowErrors_str,
    }
    return force_text(result) + ending


def invalid_not_configured(
    fieldName, fieldValue, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.NOT_CONFIGURED) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "fieldValue": fieldValue,
    }


def invalid_not_reachable(
    fieldName, fieldValue, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.NOT_REACHABLE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "fieldValue": fieldValue,
    }


def invalid_not_reachable_not_configured(
    fieldName, fieldValue, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.NOT_REACHABLE_NOT_CONFIGURED) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
        "fieldValue": fieldValue,
    }


def provided(fieldName, include_error_prefix=False, error_prefix=_error_prefix):
    return Rules.get_error(Rules.PROVIDED) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "fieldName": fieldName,
    }


def schema_error_missing_attribute(
    entityName, fieldName, include_error_prefix=False, error_prefix=_error_prefix
):
    return Rules.get_error(Rules.SCHEMA_MISSING_ATTRIBUTE) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "entityName": entityName,
        "fieldName": fieldName,
    }


def file_does_not_exist(file, include_error_prefix=False, error_prefix=_error_prefix):
    return Rules.get_error(Rules.FILE_DOES_NOT_EXIST) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "file": file,
    }


def file_invalid_extension(
    extension, extensions, include_error_prefix=False, error_prefix=_error_prefix
):
    extensions_str = _separated_values_builder.build(extensions, ", ")
    return Rules.get_error(Rules.FILE_INVALID_EXTENSION) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "extension": extension,
        "extensions": extensions_str,
    }


def file_missing_extension(
    filename, extensions, include_error_prefix=False, error_prefix=_error_prefix
):
    extensions_str = _separated_values_builder.build(extensions, ", ")
    return Rules.get_error(Rules.FILE_MISSING_EXTENSION) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "filename": filename,
    }


def file_empty(file, include_error_prefix=False, error_prefix=_error_prefix):
    return Rules.get_error(Rules.FILE_EMPTY) % {
        "errorPrefix": error_prefix if include_error_prefix else "",
        "file": file,
    }


def list_duplicates(seq):
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set(x for x in seq if x in seen or seen_add(x))
    # turn the set into a list (as requested)
    return list(seen_twice)
