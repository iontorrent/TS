# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
import json
import logging

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.template import RequestContext
from requests import HTTPError

from iondb.product_integration.forms import ThermoFisherCloudConfigForm
from iondb.product_integration.models import ThermoFisherCloudAccount

logger = logging.getLogger(__name__)


@login_required
def configure(request):
    """Constructs the view for configuring your thermo fisher cloud account"""

    # handle the post to create a new tfc account
    if request.method == 'POST':
        form = ThermoFisherCloudConfigForm(data=request.POST, files=request.FILES)
        tfc_username = form.data.get('tfc_username', '')
        if not tfc_username:
            return HttpResponse(json.dumps({'error': 'You need to enter a user name.'}), content_type="application/json")

        tfcaccount = ThermoFisherCloudAccount()
        tfcaccount.username = tfc_username
        tfcaccount.user_account_id = request.user.id

        if not form.data.get('tfc_password', ''):
            return HttpResponse(json.dumps({'error': 'No password specified.'}), content_type="application/json")

        try:
            tfcaccount.setup_ampliseq(form.data['tfc_password'])
        except HTTPError as exc:
            if exc.response.status_code == 401:
                return HttpResponse(json.dumps({'error': 'Bad username and/or password.'}), content_type="application/json")
            else:
                return HttpResponse(json.dumps({'error': 'Could not link the account'}), content_type="application/json")
        except Exception as exc:
            logger.exception(exc)
            return HttpResponse(json.dumps({'error': 'Could not link the account'}), content_type="application/json")

        try:
            tfcaccount.setup_deeplaser(form.data['tfc_password'])
        except Exception as exc:
            logger.exception(exc)
            message = 'Could not setup the TFC account.'
            if hasattr(exc, "error_code"):
                if exc.error_code == "USER_ALREADY_LINKED":
                    message = "This TFC account is already linked with this TS."
            return HttpResponse(json.dumps({'error': message}), content_type="application/json")

        # verify the thermofisher account information here
        tfcaccount.save(password=form.data['tfc_password'])
        return HttpResponse(json.dumps({'error': ''}), content_type="application/json")
    else:
        form = ThermoFisherCloudConfigForm()
        ctx = RequestContext(request, {'form': form, 'error': ''})
        return render_to_response("product_integration/configure.html", context_instance=ctx)


@login_required
def delete(request, pk):
    """Constructs the view for configuring your thermo fisher cloud account"""

    tfc_account = ThermoFisherCloudAccount.objects.get(id=pk)
    try:
        tfc_account.delete()
    except Exception as exc:
        logger.exception("Exception when removing TFC account:")
        if hasattr(exc, "error_code") and exc.error_code == "PRECONDITION_FAILED":
            return HttpResponse(
                json.dumps({"error_code": exc.error_code}),
                status=412,
                content_type="application/json"
            )
        else:
            raise Exception

    return HttpResponse(json.dumps({}), content_type="application/json")
