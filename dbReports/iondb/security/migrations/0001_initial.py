# -*- coding: utf-8 -*-
from south.utils import datetime_utils as datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'SecureString'
        db.create_table(u'security_securestring', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('encrypted_string', self.gf('django.db.models.fields.CharField')(max_length=1000)),
            ('created', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('name', self.gf('django.db.models.fields.CharField')(unique=True, max_length=128)),
        ))
        db.send_create_signal(u'security', ['SecureString'])


    def backwards(self, orm):
        # Deleting model 'SecureString'
        db.delete_table(u'security_securestring')


    models = {
        u'security.securestring': {
            'Meta': {'object_name': 'SecureString'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'encrypted_string': ('django.db.models.fields.CharField', [], {'max_length': '1000'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '128'})
        }
    }

    complete_apps = ['security']