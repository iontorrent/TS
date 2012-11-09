# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.contrib.auth import logout, REDIRECT_FIELD_NAME
from django import shortcuts, template
from django.views.decorators.debug import sensitive_post_parameters
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.cache import never_cache
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.views import login
from django.http import HttpResponseRedirect, HttpResponse
import json

# no login_required
def logout_basic(request):
    """ User was just auto-logged in by REMOTE_USER.
        Log them out again via js (in template). """
    if request.user.is_authenticated():
        logout(request)
        #return 
    context = template.RequestContext(request, {})
    return shortcuts.render_to_response("rundb/logout_basic.html",
                                        context_instance=context)
@sensitive_post_parameters()
@csrf_protect
@never_cache    
def login_ajax(request, template_name='registration/login.html',
          redirect_field_name=REDIRECT_FIELD_NAME,
          authentication_form=AuthenticationForm,
          current_app=None, extra_context=None):
    result = login(request, 
                   template_name=template_name, 
                   redirect_field_name=redirect_field_name,
                   authentication_form=authentication_form, 
                   current_app=current_app, extra_context=extra_context)
    response_data = {}
    if isinstance(result, HttpResponseRedirect):
        response_data['redirect'] = result['Location']
    else:
        result.render()
        response_data['form'] = result.content
    return HttpResponse(json.dumps(response_data), mimetype="application/json")
    