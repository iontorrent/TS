# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
from django import forms


class ThermoFisherCloudConfigForm(forms.Form):
    """Form for associating an thermo fisher cloud account with the user"""

    tfc_username = forms.CharField(max_length=128, label="Username/Email")
    tfc_password = forms.CharField(widget=forms.PasswordInput(), label="Password", required=False)
