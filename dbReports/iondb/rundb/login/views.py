# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import json
import urlparse
from django.contrib.auth import REDIRECT_FIELD_NAME, login as auth_login
from django.contrib.sites.models import get_current_site
from django import shortcuts, template
from django.core import urlresolvers
from django.views.decorators.debug import sensitive_post_parameters
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.cache import never_cache
from django.contrib.auth.models import User, Group
from django.contrib.auth.views import logout_then_login
from django.http import HttpResponseRedirect, HttpResponse

from iondb.rundb.login.forms import UserRegistrationForm, AuthenticationRememberMeForm
from iondb.rundb.models import Message
from django.conf import settings
from django.template.response import TemplateResponse

import logging
logger = logging.getLogger(__name__)

# No login_required
def logout_view(request):
    return logout_then_login(request)


@sensitive_post_parameters()
#@csrf_protect
@never_cache
def remember_me_login(request, template_name='registration/login.html',
          redirect_field_name=REDIRECT_FIELD_NAME,
          authentication_form=AuthenticationRememberMeForm,
          current_app=None, extra_context=None):
    """
    Displays the login form and handles the login action.
    """
    redirect_to = request.REQUEST.get(redirect_field_name, '')

    if request.method == "POST":
        form = authentication_form(data=request.POST)
        if form.is_valid():
            netloc = urlparse.urlparse(redirect_to)[1]

            # Use default setting if redirect_to is empty
            if not redirect_to:
                redirect_to = settings.LOGIN_REDIRECT_URL

            # Heavier security check -- don't allow redirection to a different
            # host.
            elif netloc and netloc != request.get_host():
                redirect_to = settings.LOGIN_REDIRECT_URL

            ##### BEGIN REMEMBER ME #######
            if not form.cleaned_data.get('remember_me'):
                logger.debug('setting session expiry 0')
                request.session.set_expiry(0)
            else:
                logger.debug('setting session expiry %s' % settings.SESSION_COOKIE_AGE)
                request.session.set_expiry(settings.SESSION_COOKIE_AGE)
            ##### END REMEMBER ME #######

            # Okay, security checks complete. Log the user in.
            auth_login(request, form.get_user())

            if request.session.test_cookie_worked():
                request.session.delete_test_cookie()

            return HttpResponseRedirect(redirect_to)
    else:
        form = authentication_form(request)

    request.session.set_test_cookie()

    current_site = get_current_site(request)
    browser = request.META['HTTP_USER_AGENT']

    context = {
        'form': form,
        redirect_field_name: redirect_to,
        'site': current_site,
        'site_name': current_site.name,
        'prompt_chromeframe': 'MSIE' in browser and 'chromeframe' not in browser
    }
    if extra_context is not None:
        context.update(extra_context)
    return TemplateResponse(request, template_name, context,
                            current_app=current_app)

@sensitive_post_parameters()
#@csrf_protect
@never_cache
def login_ajax(request, template_name='registration/login.html',
               redirect_field_name=REDIRECT_FIELD_NAME,
               authentication_form=AuthenticationRememberMeForm,
               current_app=None, extra_context=None):
    result = remember_me_login(request,
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


# NO login_required - new users
@sensitive_post_parameters()
@csrf_protect
def registration(request):
    if request.method == 'POST':  # If the form has been submitted...
        form = UserRegistrationForm(request.POST)
        if form.is_valid():  # All validation rules pass
            # create user from cleaned_data
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password1']

            new_user = User.objects.create_user(username, email, password)
            new_user.is_active = False  # Users created inactive by default

            # Add users to ionusers group, for a default set of permissions
            try:
                group = Group.objects.get(name='ionusers')
                new_user.groups.add(group)
            except Group.DoesNotExist:
                logger.warn("Group ionusers not found. " +
                         "New user %s will lack permission to do anything beyond look!", username)
            new_user.save()

            # Send global message notifying of pending registration
            msg = "New pending user registration for '%s'. " + \
                "Please visit <a href='%s'>Account Management</a> (as an admin user) to review."
            Message.warn(msg % (username, urlresolvers.reverse('configure_account')))

            # Otherwise redirect to a success page. Awaiting approval
            return shortcuts.redirect(urlresolvers.reverse('signup_pending'))
    else:
        # Blank Form
        form = UserRegistrationForm()

    context = template.RequestContext(request, {'form': form, })
    return shortcuts.render_to_response("rundb/login/ion_account_reg.html",
                                        context_instance=context)
