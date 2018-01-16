# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import

from iondb.product_integration.models import ThermoFisherCloudAccount
from django.contrib import admin
import logging

logger = logging.getLogger(__name__)


class ThermoFisherCloudAccountAdmin(admin.ModelAdmin):
    """Admin interface for """
    list_display = ('username', 'deeplaser_principleid')
    list_filter = ('username', 'deeplaser_principleid')
    search_fields = ['username', 'deeplaser_principleid']

    def has_add_permission(self, request):
        return False

# register the admin interfaces
admin.site.register(ThermoFisherCloudAccount, ThermoFisherCloudAccountAdmin)
