# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.http import HttpResponse, HttpResponsePermanentRedirect
from django.contrib.auth.decorators import login_required
from django.conf import settings

import os.path

# from mimetypes import guess_type

"""
Serve up local content via wsgi Location redirect
Allows us to enforce login_required,
But then uses apache to serve content

Approach adapted from: https://github.com/johnsensible/django-sendfile/
"""


def serve_wsgi_location(request, urlpath, **kwargs):
    webroot = getattr(settings, "SERVE_LOCATION_ROOT", "/var/www")
    filename = os.path.join(webroot, urlpath)

    # Extra stat, but gives us a django 404 page instead of Apache /private/ error.
    if not os.path.exists(filename):
        from django.http import Http404

        raise Http404('"%s" does not exist' % filename)

    if os.path.isdir(filename) and not urlpath.endswith("/"):
        return HttpResponsePermanentRedirect("/%s/" % urlpath)

    # Propagate query_string (to php scripts in output)
    args = request.META.get("QUERY_STRING", "")
    if args:
        urlpath = "%s?%s" % (urlpath, args)

    response = HttpResponse()

    # WSGI internal redirect to /private/ (see torrent-server.conf)
    response["Location"] = "/private/" + urlpath

    # need to destroy get_host() to stop django
    # rewriting our location to include http, so that
    # mod_wsgi is able to do the internal redirect
    request.get_host = lambda: ""

    # Files in download_links should always be offered for download, not shown inline
    if "/download_links/" in urlpath:
        attachment_filename = kwargs.get("attachment_filename") or os.path.basename(
            filename
        )
        response["Content-Disposition"] = (
            'attachment; filename="%s"' % attachment_filename
        )

    # HttpResponse sets a default mime type of text/html,
    # which is not appropriate here.
    del response["Content-Type"]
    del response["Content-Encoding"]

    # If omitted entirely, apache/wsgi is not setting a correct one either.
    # mimetype, encoding = guess_type(filename)
    # response['Content-Type'] = mimetype or 'application/octet-stream'
    # if encoding:
    #    response['Content-Encoding'] = encoding

    return response
