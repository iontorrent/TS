"""
this is the scenario that we support where `oem_support_settings.py` will turn
the disabled feature back on for support purpose.

```
    from iondb.oem_support_settings import OEMSUPPORT_FEATURE_FLAGS 
    if OEMSUPPORT_FEATURE_FLAGS:
        OEM_FEATURE_FLAGS = OEMSUPPORT_FEATURE_FLAGS + OEM_FEATURE_FLAGS
```

All flags are turned on by default. The addition will not have any effect.

>>> f1 = FeatureFlags()
>>> f2 = FeatureFlags()
>>> f = f1 + f2
>>> f1.NEWS
True
>>> f2.NEWS
True
>>> f.NEWS
True
>>> f1 == f2
True
>>> f1 == f
True
>>> f2 == f
True

When one of the flag is set to False and other True, the result will be turned on.

>>> f1 = FeatureFlags()
>>> f2 = FeatureFlags()
>>> f2.NEWS = False
>>> f = f1 + f2
>>> f1.NEWS
True
>>> f2.NEWS
False
>>> f.NEWS
True
>>> f1 == f2
False
>>> f1 == f
True
>>> f2 == f
False

the order of addition should not matter either:

>>> f = f2 + f1
>>> f1.NEWS
True
>>> f2.NEWS
False
>>> f.NEWS
True
>>> f1 == f2
False
>>> f1 == f
True
>>> f2 == f
False


When flag in both objects are off, the result remains unchanged.

>>> f1 = FeatureFlags()
>>> f1.NEWS = False
>>> f2 = FeatureFlags()
>>> f2.NEWS = False
>>> f = f1 + f2
>>> f1.NEWS
False
>>> f2.NEWS
False
>>> f.NEWS
False
>>> f1 == f2
True
>>> f1 == f
True
>>> f2 == f
True

Intended scenario. Set OEM flags to be off and turn one of them on in oem_support.py

>>> oem_default = FeatureFlags(default_on=False)
>>> oem_support = FeatureFlags(default_on=False)
>>> oem_support.BARCODESET_ACTIONS = True
>>> oem_default = oem_support + oem_default
>>> oem_default.NEWS
False
>>> oem_default.BARCODESET_ACTIONS
True

"""


class FeatureFlags:
    """
    default to true and can be changed after created
    
    >>> f = FeatureFlags()
    >>> f.AMPLISEQ
    True
    >>> f.AMPLISEQ = False
    >>> f.AMPLISEQ
    False

    set to false when initialized

    >>> f1 = FeatureFlags(default_on=False)
    >>> f1.AMPLISEQ
    False
    >>> f1.AMPLISEQ = True
    >>> f1.AMPLISEQ
    True
    
    """

    IONREPORTERUPLOADER = True
    AMPLISEQ = True
    IMPORT_PANEL_ARCHIVE = True
    CONFIGURE_ANALYSISPARAMETERS = True
    OFFCYCLE_UPDATES = True
    CONFIGURE_SERVICES = True
    REPORT_METAL = True
    NEWS = True
    SENDFEEDBACK = True
    HELP_ION_DOCS = True
    PLAN_TEMPLATE_RESEARCHAPPLICATIONS = True
    DATA_RESULTS_COMBINE_SELECTED = True
    DATA_RESULTS_COMPARE_ALL = True
    DATA_DOWNLOAD_FILTERED_RESULTS_AS_CSV = True
    DATA_DOWNLOAD_FILTERED_RESULTS_SELECTED_AS_CSV = True
    REPORT_DETAILS_SECTION = True
    REPORTS_REPORT_PDF = True
    REPORTS_PLUGIN_PDF = True
    CONFIGURE_REFERENCES_REFERENCES_IMPORT_PRELOADED_ION_REFERENCES = True
    CONFIGURE_REFERENCES_OBSOLETEREFERENCES = True
    PUBLISHERS_ACTIONS = True
    BARCODESET_ACTIONS = True
    TESTFRAGMENT_ACTIONS = True
    NAS_STORAGE = True
    PLUGINS_PLUGIN_EDIT_SELECTED = True
    PLUGINS_PLUGIN_EDIT_DEFAULTSELECTED = True
    PLUGINS_PLUGIN_DISPLAY_ISSUPPORTED = True
    PLUGINS_PLUGIN_INSTALL_DETAILED_ERRORMSG = True
    PLUGINS_PLUGIN_INSTALLTOVERSION_DETAILED_ERRORMSG = True
    PLUGINS_PLUGIN_UNINSTALL_DETAILED_ERRORMSG = True
    PLUGINS_PLUGIN_UPGRADE_DETAILED_ERRORMSG = True
    HELP_SDK_DOCUMENATION = True
    HELP_SEQUENCING_PROTOCOLS = True
    DOWNLOAD_CSA = True
    LOGIN_IONTORRENT_WORKFLOWS = True

    def __init__(self, default_on=True):
        self._attributes = [
            attr for attr in FeatureFlags.__dict__.keys() if not attr.startswith("__")
        ]
        for attr in self._attributes:
            setattr(self, attr, default_on)

    def __add__(self, other):
        new_featureflags = FeatureFlags()
        for attr in self._attributes:
            flag = getattr(self, attr) or getattr(other, attr)
            setattr(new_featureflags, attr, flag)
        return new_featureflags

    def __eq__(self, other):
        for attr in self._attributes:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True


if __name__ == "__main__":
    import doctest

    doctest.testmod()
