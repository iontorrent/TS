# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
import json
import logging

from django.conf.urls import url
from django.http import HttpResponse
from tastypie.authorization import DjangoAuthorization
from tastypie.resources import ModelResource
from tastypie.utils import trailing_slash

from iondb.product_integration.models import DeepLaserResponse
from iondb.product_integration.utils import send_deep_laser_iot_request, send_deep_laser_device_response
from iondb.product_integration.tasks import handle_deep_laser_device_request
from iondb.rundb.authn import IonAuthentication

logger = logging.getLogger(__name__)


class DeepLaserResponseResource(ModelResource):
    MAX_RESPONSES_STORED = 1000

    def prepend_urls(self):
        urls = [
            url(r"^(?P<resource_name>%s)/send_iot_request%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('send_iot_request')),
            url(r"^(?P<resource_name>%s)/receive_iot_response%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('receive_iot_response')),
            url(r"^(?P<resource_name>%s)/receive_device_request%s$" % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('receive_device_request')),
        ]
        return urls

    def send_iot_request(self, request, **kwargs):
        """ This url is for other apps to call to send deep laser requests. The CloudUpload plugin would hit this."""
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        request_json_object = json.loads(request.body)
        send_deep_laser_iot_request(request_json_object)

        return HttpResponse()

    def receive_iot_response(self, request, **kwargs):
        """ Deep Laser POSTS to this url with responses to requests sent above. """
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        request_json_object = json.loads(request.body)

        request_id = request_json_object["requestId"]
        request_type = request_json_object["requestType"]
        request_status_code = int(request_json_object["statusCode"])

        request_response = request_json_object["response"]

        logger.debug("Received async iot response from DL request_id=%s\n%s\n",
                     request_id,
                     json.dumps(request_json_object, sort_keys=True, indent=2))

        for response in DeepLaserResponse.objects.order_by("-date_received")[self.MAX_RESPONSES_STORED:]:
            response.delete()

        response_model, created = DeepLaserResponse.objects.get_or_create(request_id=request_id, defaults={
            "request_type": request_type,
            "response": request_response,
            "status_code": request_status_code
        })

        if not created:
            response_model.request_type = request_type
            response_model.response = request_response
            response_model.status_code = request_status_code

        response_model.save()

        logger.debug("Created model with pk=%i", response_model.pk)
        return HttpResponse()

    def receive_device_request(self, request, **kwargs):
        """ Deep Laser POSTS to this url with commands from the TFC dashboard.
        We then need to ack them with an http POST back to DL. """
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        request_json_object = json.loads(request.body)
        request_key = request_json_object["key"]
        request_type = request_json_object["type"]
        request_parameters = request_json_object["parameters"]
        request_principle_id = request_json_object.get("principalId")

        logger.debug("Received async device request from DL \n%s\n",
                     json.dumps(request_json_object, sort_keys=True, indent=2))

        device_request_succeeded = True
        try:
            handle_deep_laser_device_request(request_key, request_parameters, request_principle_id)
        except Exception as exc:
            device_request_succeeded = False
            logger.exception("Exception when calling handle_deep_laser_device_request:")

        if request_type == "DEVICE":
            if device_request_succeeded:
                send_deep_laser_device_response({
                    "status": "SUCCESS",
                    "result": {}
                })
            else:
                send_deep_laser_device_response({
                    "status": "ERROR",
                    "result": {}
                })

        return HttpResponse()

    def dehydrate(self, bundle):
        """ Special case for the "linkuser" response. It contains sensitive certs we don't want in the api. """
        if bundle.data["request_type"] == "linkuser":
            bundle.data["response"]["certificateMetadata"] = None
            bundle.data["response"]["principalId"] = None
        return bundle

    class Meta:
        queryset = DeepLaserResponse.objects.all()
        authentication = IonAuthentication()
        authorization = DjangoAuthorization()
