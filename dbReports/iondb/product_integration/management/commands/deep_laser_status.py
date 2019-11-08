from django.core.management.base import BaseCommand
from iondb.rundb.models import GlobalConfig
from iondb.product_integration.utils import (
    send_deep_laser_iot_request,
    get_deep_laser_iot_response,
)
import json


class Command(BaseCommand):
    help = "Reach out to Deep Laser and see if we are connected."

    def handle(self, *args, **options):
        # Get GC status
        gc = GlobalConfig.objects.get()
        telemetry_enabled = gc.telemetry_enabled
        self.stdout.write(
            "Telemetry enabled in Global Config: {}".format(telemetry_enabled)
        )

        # Do DL heartbeat
        self.stdout.write(
            "Sending 'tfcheartbeat' request to Deep Laser and waiting for response..."
        )
        try:
            request_id = send_deep_laser_iot_request({"request": "tfcheartbeat"})
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(
                    "Could not POST the request to Deep Laser! Is it running?"
                )
            )
            raise e
        try:
            response_object = get_deep_laser_iot_response(request_id, timeout=30)
        except ValueError:
            self.stdout.write(
                self.style.ERROR("Timeout while waiting for DL response!")
            )
            return False

        self.stdout.write(
            "Got response from Deep Laser: {}".format(
                json.dumps(response_object.response)
            )
        )

        if response_object.response.get("connected") == "True":
            self.stdout.write("Deep Laser connection is OKAY!")
        else:
            self.stdout.write(self.style.ERROR("Deep Laser connection is DOWN!"))
