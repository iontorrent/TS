import os
import sys
import subprocess
import json
import requests
import sys

try:
    sys.path.append("/opt/ion/")
    os.environ["DJANGO_SETTINGS_MODULE"] = "iondb.settings"
    from django.conf import settings
except ImportError as Err:
    print(Err)
    sys.exit(0)
except Exception as Err:
    print(Err)
    sys.exit(0)

def getMajorPlatform():
    try:
        from iondb.rundb.models import GlobalConfig
        majorPlatform = GlobalConfig.get().majorPlatform
        return majorPlatform
    except Exception as Err:
        return "NOT_AVAILABLE"

# Use local deprecation offcyle json path if available
def get_deprecation_messages_local(localpath):
    with open(localpath) as json_file:
        return json.load(json_file)
    return None

# Look for local deprecation json first then ionupdates.com
def get_deprecation_json():
    depreOffcyleLocal = os.path.join(settings.OFFCYCLE_UPDATE_PATH_LOCAL, "deprecation_data.json")
    if os.path.exists(depreOffcyleLocal):
        with open(depreOffcyleLocal, 'r') as fh:
            return json.load(fh)

    TIMEOUT_LIMIT_SEC = settings.REQUESTS_TIMEOUT_LIMIT_SEC
    try:
        resp = requests.get(settings.OFFCYLE_DEPRECATION_MSG, timeout=TIMEOUT_LIMIT_SEC)
        resp.raise_for_status()
        return resp.json()
    except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as err:
        print (
            "get_deprecation_messages timeout or connection errors for {u}: {e}".format(
                e=str(err), u=settings.OFFCYLE_DEPRECATION_MSG
            )
        )
        return None
    except ValueError as decode_err:
        print("get_deprecation_messages JSON decode error: {}".format(str(decode_err)))
        return None

if __name__ == "__main__":
    # check for root level permissions
    if os.geteuid() != 0:
        sys.exit(
            "You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting."
        )
    if (len(sys.argv) == 1):
        # Hanle major platform and deprecation status not available
        try:
            enableSwitchRepo = False
            deprecationMessages = get_deprecation_json()
            if deprecationMessages:
                enableSwitchRepo = deprecationMessages.get("enableDeprecationMessage")
            print("{0} {1}".format(getMajorPlatform(), enableSwitchRepo))
        except Exception as err:
            print("{0} {1}".format("NOT_AVAILABLE", False))

