# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from django.conf import settings

from django.conf.urls import patterns, url, include

from django.contrib.staticfiles.urls import staticfiles_urlpatterns


# Uncomment the next two lines to enable the admin:
from django.contrib import admin

admin.autodiscover()
# admin.site.login_template = "rundb/login.html"

from iondb.rundb.login.urls import urlpatterns as login_patterns
from iondb.servelocation import serve_wsgi_location
from iondb.utils.utils import is_TsVm

js_info_dict = {
    "domain": "djangojs",  # using the djangojs Message File, which is different from the Message File used by server-side django code (views, templates & .py)
    # set packages to restrict to not pull in all of Django
    # 'packages': ('iondb.ftpserver',
    #             'iondb.rundb',
    #             'iondb.security',
    #             'iondb.product_integration',),
    "packages": None,
}

urlpatterns = patterns(
    r"",
    #  TODO: i18n Update apache to not require context prefix of configure/
    #  TODO: i18n Performance use caching for django.views.i18n.javascript_catalog
    (r"^configure/jsi18n/$", "django.views.i18n.javascript_catalog", js_info_dict),
    (r"^product_integration/", include("iondb.product_integration.urls")),
    (r"^configure/", include("iondb.rundb.configure.urls")),
    (r"^data/", include("iondb.rundb.data.urls")),
    (r"^monitor/", include("iondb.rundb.monitor.urls")),
    (r"^plan/", include("iondb.rundb.plan.urls")),
    (r"^sample/", include("iondb.rundb.sample.urls")),
    (r"^report/", include("iondb.rundb.report.urls")),
    (r"^rundb/", include("iondb.rundb.urls")),
    (r"^security/", include("iondb.security.urls")),
    (r"^home/", include("iondb.rundb.home.urls")),
    # Admin
    (r"^admin/doc/", include("django.contrib.admindocs.urls")),
    (r"^admin/manage/$", "iondb.rundb.admin.manage"),
    (r"^admin/network/$", "iondb.rundb.admin.network"),
    (
        r"^admin/network/how_is/(?P<host>[\w\.]+):(?P<port>\d+)/feeling$",
        "iondb.rundb.admin.how_are_you",
    ),
    (r"^admin/update/$", "iondb.rundb.admin.update"),
    (r"^admin/update/check/$", "iondb.rundb.admin.run_update_check"),
    (r"^admin/update/switch_repo/(?P<majorPlatform>[\w\.]+)", "iondb.rundb.admin.switch_repo"),
    (r"^admin/update/tsconfig_log/$", "iondb.rundb.admin.tsconfig_log"),
    (r"^admin/update/download_logs/$", "iondb.rundb.admin.get_zip_logs"),
    (r"^admin/updateOneTouch/$", "iondb.rundb.admin.updateOneTouch"),
    (r"^admin/updateOneTouch/ot_log$", "iondb.rundb.admin.ot_log"),
    (r"^admin/update/install_log$", "iondb.rundb.admin.install_log"),
    (r"^admin/update/install_lock$", "iondb.rundb.admin.install_lock"),
    (
        r"^admin/update/version_lock/(?P<enable>[\w\.]+)",
        "iondb.rundb.admin.version_lock",
    ),
    (r"^admin/update/maintenance/(?P<action>[\w\.]+)", "iondb.rundb.admin.maintenance"),
    (
        r"^admin/experiment/exp_redo_from_scratch/$",
        "iondb.rundb.admin.exp_redo_from_scratch",
    ),
    url(
        r"^admin/configure_server/$",
        "iondb.rundb.admin.configure_server",
        name="configure_server",
    ),
    url(r"^admin/tsvm/$", "iondb.rundb.admin.tsvm_control", name="tsvm"),
    url(
        r"^admin/tsvm/(?P<action>\w+)/$", "iondb.rundb.admin.tsvm_control", name="tsvm"
    ),
    url(r"^admin/tsvm_log/(.+)/$", "iondb.rundb.admin.tsvm_get_log", name="tsvm_log"),
    # password change doesn't accept extra_context
    (r"^admin/password_change/done/", admin.site.password_change_done),
    (r"^admin/password_change/", "iondb.rundb.admin.configure_account_admin"),
    (r"^admin/logout/", admin.site.logout),

    (r"^admin/$", admin.site.index, {"extra_context": {"is_VM": is_TsVm()}}),
    (
        r"^admin/(?P<app_label>\w+)/$",
        admin.site.app_index,
        {"extra_context": {"is_VM": is_TsVm()}},
    ),
    (r"^admin/", include(admin.site.urls)),
    (r"^(?P<urlpath>output.*)$", serve_wsgi_location),
    (r"^(?P<urlpath>chef_logs.*)$", serve_wsgi_location),
    (r"^(?P<urlpath>ot_logs.*)$", serve_wsgi_location),
)
urlpatterns.extend(login_patterns)
urlpatterns.extend(staticfiles_urlpatterns())

if settings.TEST_INSTALL:
    from os import path

    urlpatterns.extend(
        patterns(
            "",
            (
                r"^site_media/(?P<path>.*)$",
                "django.views.static.serve",
                {"document_root": settings.MEDIA_ROOT},
            ),
            (
                r"^testreports/(?P<path>.*)$",
                "django.views.static.serve",
                {
                    "document_root": path.join(
                        path.dirname(settings.MEDIA_ROOT), "testreports"
                    )
                },
            ),
        )
    )
