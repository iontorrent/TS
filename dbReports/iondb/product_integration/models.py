# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
"""
NOTE: The GPG algorithm is sensitive to the encoding so if the strings are encoded in something
different than what the django locale's preferred encoding then it will fail to decrypt
"""

import requests
import os
from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver

from iondb.product_integration.utils import send_deep_laser_iot_request, get_deep_laser_iot_response
from iondb.rundb.json_field import JSONField
from iondb.security.models import SecureString
from iondb import settings


class ThermoFisherCloudAccount(models.Model):
    """This class will wrap up all of the credential issues for hooking into the thermo fisher cloud"""

    # username (or email) for the TFC account
    username = models.CharField(null=False, max_length=128, unique=True, blank=False)

    # the deeplaser principle id
    deeplaser_principleid = models.CharField(null=True, max_length=100)

    # foreign key to the user account
    user_account = models.ForeignKey(User)

    @property
    def ampliseq_secret_name(self):
        """Gets the name of the secret for saving and retrieval"""
        return "ampliseq_" + self.user_account.username + "_" + self.username

    def remove_ampliseq(self):
        """Removes the ampliseq secure string for the password"""
        try:
            secure_string = SecureString.objects.get(name=self.ampliseq_secret_name)
            secure_string.delete()
        except SecureString.DoesNotExist:
            pass

    def setup_deeplaser(self, password):
        """Setup a connection with deeplaser"""
        pass
        # request_id = send_deep_laser_iot_request({
        #     "request": "linkuser",
        #     "username": self.username,
        #     "password": password
        # })
        # response = get_deep_laser_iot_response(request_id, timeout=15)
        # if response.status_code == 200:
        #     self.deeplaser_principleid = response.response["principalId"]
        # else:
        #     exc = ValueError("Bad response from deep laser!")
        #     exc.error_code = response.response.get("errorCode")
        #     raise exc

    def remove_deeplaser(self):
        """Removes the object from deep laser"""
        pass
        # request_id = send_deep_laser_iot_request({
        #     "request": "unlinkuser",
        #     "principalId": self.deeplaser_principleid,
        # })
        # response = get_deep_laser_iot_response(request_id, timeout=15)
        # if response.status_code != 200:
        #     exc = ValueError("Bad response from deep laser!")
        #     exc.error_code = response.response.get("errorCode")
        #     if exc.error_code == "USER_NOT_LINKED":
        #         # This error code can mean that a user has already unlinked on the other end.
        #         # We are going to assume the is the case and just say the unlinked has succeeded.
        #         pass
        #     elif exc.error_code == "MQTT_CONNECTION_ERROR":
        #         raise exc
        #     else:
        #         raise exc

    def setup_ampliseq(self, password):
        """This method will setup a secret for the ampliseq password"""
        base_url = os.path.join(settings.AMPLISEQ_URL,"ws/design/list")
        ampliseq_response = requests.get(base_url, auth=(self.username, password))
        ampliseq_response.raise_for_status()

    def get_ampliseq_password(self):
        """This will query the secure storage for the secret name and return the password unencrypted"""
        secret = SecureString.objects.get(name=self.ampliseq_secret_name)
        return secret.decrypted

    def save(self, *args, **kwargs):
        """Override the default save behavior to also include the secure secret"""
        password = kwargs.pop('password', '')
        super(ThermoFisherCloudAccount, self).save(*args, **kwargs)
        if password:
            secret = SecureString.create(password, self.ampliseq_secret_name)
            secret.save()


@receiver(pre_delete, sender=ThermoFisherCloudAccount, dispatch_uid="delete_tfcaccount")
def on_pluginresult_delete(sender, instance, **kwargs):
    """Delete all of the files for a pluginresult record """
    instance.remove_deeplaser()
    instance.remove_ampliseq()


class DeepLaserResponse(models.Model):
    """This model store iot responses from deep laser"""
    date_received = models.DateTimeField(auto_now_add=True)
    request_id = models.CharField(max_length=512, db_index=True)
    request_type = models.CharField(max_length=512, db_index=True)
    response = JSONField()
    status_code = models.PositiveSmallIntegerField()

    class Meta:
        ordering = ['-date_received']
