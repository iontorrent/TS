# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf import settings
try:
    from django.conf.urls import patterns, url, include
except ImportError:
    # Compat Django 1.4
    from django.conf.urls.defaults import patterns, url, include

from django.contrib.staticfiles.urls import staticfiles_urlpatterns


# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()
# admin.site.login_template = "rundb/login.html"

from iondb.rundb.login.urls import urlpatterns as login_patterns
from iondb.servelocation import serve_wsgi_location
from iondb.utils.utils import is_TsVm

urlpatterns = patterns(
    r'',
    (r'^configure/', include('iondb.rundb.configure.urls')),
    (r'^data/', include('iondb.rundb.data.urls')),
    (r'^monitor/', include('iondb.rundb.monitor.urls')),
    (r'^plan/', include('iondb.rundb.plan.urls')),
    (r'^sample/', include('iondb.rundb.sample.urls')),
    (r'^report/', include('iondb.rundb.report.urls')),
    (r'^rundb/', include('iondb.rundb.urls')),
    (r'^security/', include('iondb.security.urls')),
    # From extra/
    url(r'^news/$', 'iondb.rundb.extra.views.news', name="news"),
    # Admin
    (r'^admin/doc/', include('django.contrib.admindocs.urls')),
    (r'^admin/manage/$', 'iondb.rundb.admin.manage'),
    (r'^admin/network/$', 'iondb.rundb.admin.network'),
    (r'^admin/network/check$', 'iondb.rundb.admin.how_am_i'),
    (r'^admin/network/external_ip/$', 'iondb.rundb.admin.fetch_remote_content', {"url": settings.EXTERNAL_IP_URL}),
    (r'^admin/network/how_is/(?P<host>[\w\.]+):(?P<port>\d+)/feeling$', 'iondb.rundb.admin.how_are_you'),
    (r'^admin/update/$', 'iondb.rundb.admin.update'),
    (r'^admin/update/check/$', 'iondb.rundb.admin.run_update_check'),
    (r'^admin/update/tsconfig_log/$', 'iondb.rundb.admin.tsconfig_log'),
    (r'^admin/update/download_logs/$', 'iondb.rundb.admin.get_zip_logs'),
    (r'^admin/updateOneTouch/$', 'iondb.rundb.admin.updateOneTouch'),
    (r'^admin/updateOneTouch/ot_log$', 'iondb.rundb.admin.ot_log'),
    (r'^admin/update/install_log$', 'iondb.rundb.admin.install_log'),
    (r'^admin/update/install_lock$', 'iondb.rundb.admin.install_lock'),
    (r'^admin/update/version_lock/(?P<enable>[\w\.]+)', 'iondb.rundb.admin.version_lock'),
    (r'^admin/update/maintenance/(?P<action>[\w\.]+)', 'iondb.rundb.admin.maintenance'),
    (r'^admin/experiment/exp_redo_from_scratch/$', 'iondb.rundb.admin.exp_redo_from_scratch'),
    url(r'^admin/tsvm/$', 'iondb.rundb.admin.tsvm_control', name="tsvm"),
    url(r'^admin/tsvm/(?P<action>\w+)/$', 'iondb.rundb.admin.tsvm_control', name="tsvm"),
    url(r'^admin/tsvm_log/(.+)/$', 'iondb.rundb.admin.tsvm_get_log', name="tsvm_log"),

    # password change doesn't accept extra_context
    (r'^admin/password_change/done/', admin.site.password_change_done),
    (r'^admin/password_change/', admin.site.password_change),
    (r'^admin/logout/', admin.site.logout),
    (r'^admin/$', admin.site.index, {'extra_context': {'is_VM': is_TsVm() }}),
    (r'^admin/(?P<app_label>\w+)/$', admin.site.app_index, {'extra_context': {'is_VM': is_TsVm() }}),
    (r'^admin/', include(admin.site.urls)),

    (r'^(?P<urlpath>output.*)$', serve_wsgi_location),
    (r'^(?P<urlpath>chef_logs.*)$', serve_wsgi_location),
    (r'^(?P<urlpath>ot_logs.*)$', serve_wsgi_location),
)
urlpatterns.extend(login_patterns)
urlpatterns.extend(staticfiles_urlpatterns())

if settings.TEST_INSTALL:
    from os import path
    urlpatterns.extend(patterns(
        '',
        (r'^site_media/(?P<path>.*)$',
         'django.views.static.serve',
         {'document_root': settings.MEDIA_ROOT}),
        (r'^testreports/(?P<path>.*)$',
         'django.views.static.serve',
         {'document_root': path.join(path.dirname(settings.MEDIA_ROOT),
                                     "testreports")})))
