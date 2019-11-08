#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

import os
import json
import argparse


def remove_entry(id_server, fpname):
    """
    Remove an entry from the file:
    Loop thru each entry and add entry to new list if it does not match id.
    Overwrite file with the new list.
    """
    if os.path.isfile(fpname):
        newlist = list()
        with open(fpname, "rb") as fh:
            data = json.load(fh)
        for appliance in data.get("appliances"):
            if appliance.get("ipaddress") == id_server:
                pass
            else:
                newlist.append(appliance)

        if len(newlist) > 0:
            data = {"appliances": newlist}
            with open(fpname, "wb") as fh:
                json.dump(data, fh)
        else:
            os.remove(fpname)
    else:
        print("File does not exist: %s" % fpname)


def edit_entry(id_server, id_user, password, fpname):
    """Add/Modify an entry in the file"""
    newappliance = {"username": id_user, "password": password, "ipaddress": id_server}
    if os.path.isfile(fpname):
        with open(fpname, "rb") as fh:
            data = json.load(fh)
        updated = False
        for appliance in data.get("appliances"):
            if appliance.get("ipaddress") == id_server:
                updated = True
                appliance["username"] = id_user
                appliance["password"] = password
        if not updated:
            data["appliances"].append(newappliance)
    else:
        data = {"appliances": [newappliance]}

    with open(fpname, "wb") as fh:
        json.dump(data, fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--id", required=True)
    parser.add_argument("--us")
    parser.add_argument("--pw")
    args = parser.parse_args()

    servername = args.id
    username = args.us if args.us else "admin"
    cred = args.pw if args.pw else "nexenta"
    nexentacred = "/etc/torrentserver/nms_access"

    if args.remove:
        remove_entry(servername, nexentacred)
    else:
        edit_entry(servername, username, cred, nexentacred)
