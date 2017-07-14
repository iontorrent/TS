# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import

from iondb.security.models import SecureString
from django.contrib import admin
import logging

logger = logging.getLogger(__name__)


class SecureStringAdmin(admin.ModelAdmin):
    """Admin interface for """
    list_display = ('created', 'name')
    list_filter = ('created', 'name')
    search_fields = ['created', 'name']
    ordering = ('-created', 'name')
    exclude = ('encrypted_string', )

    def has_add_permission(self, request):
        return False

# register the admin interfaces
admin.site.register(SecureString, SecureStringAdmin)
