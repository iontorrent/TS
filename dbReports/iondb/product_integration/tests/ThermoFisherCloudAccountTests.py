# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from iondb.product_integration.models import ThermoFisherCloudAccount


class ThermoFisherCloudAccountTest(TestCase):
    """This will test the thermo fisher cloud account model"""

    def test_init(self):
        """Tests the constructor"""
        ThermoFisherCloudAccount("MyUserName")
