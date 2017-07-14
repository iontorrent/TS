# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
"""This script must be run in the Torrent Server environment and takes a
barcode set name and the path to a barcode csv file in the same format
uploaded to the Torrent Server.
"""


__author__ = "Brian Kennedy"
from iondb.bin.djangoinit import *
from iondb.rundb.configure.views import *
from iondb.rundb.configure.views import _validate_barcode

import sys


def _get_new_or_db_barcode(name, index):
    if (name and index):
        barcodes = dnaBarcode.objects.filter(name = name, index = index)
        if barcodes:
            print ("db barcode FOUND for name=%s; index=%s" %(name, index))
            return barcodes[0]
    return dnaBarcode()

def command_line_add_barcode(name, dest_path, is_override):
    """add the barcodes, with CSV validation"""

    # name = request.POST.get('name', '')
    # postedfile = request.FILES['postedfile']
    # destination = tempfile.NamedTemporaryFile(delete=False)
    # for chunk in postedfile.chunks():
    #    destination.write(chunk)
    # postedfile.close()
    # destination.close()
    # check to ensure it is not empty
    headerCheck = open(dest_path, "rU")
    firstCSV = []
    for firstRow in csv.reader(headerCheck):
        firstCSV.append(firstRow)
    headerCheck.close()
    if not firstRow:
        print("Error: Barcode file is empty")
        return
    expectedHeader = ["id_str", "type", "sequence", "floworder",
        "index", "annotation", "adapter", "score_mode", "score_cutoff"]
    ##if sorted(firstCSV[0]) != sorted(expectedHeader):
    ##    print("Barcode csv header is not as expected. Please try again starting with the provided example")
    ##    return

    # test if the barcode set name has been used before
    barCodeSet = dnaBarcode.objects.filter(name=name)
    if barCodeSet and not is_override:
        print("Nothing to do: Barcode set with the same name %s already exists" %name)
        return True

    is_override_option = is_override
    if is_override and not barCodeSet:
        is_override_option = False
        
    print(">>> Start processing barcode set %s" %name)
    index = 0
    barCodes = []
    failed = {}
    file = open(dest_path, "rU")
    reader = csv.DictReader(file)
    for index, row in enumerate(reader, start=1):
        invalid = _validate_barcode(row)
        if invalid:  # don't make dna object or add it to the list
            failed[index] = invalid
            continue
        
        newBarcode = _get_new_or_db_barcode(name, index)

        newBarcode.name = name  # set the name
        newBarcode.index = index  # set index this can be overwritten later
        nucs = ["sequence", "floworder", "adapter"]  # fields that have to be uppercase
        for key, value in row.items():  # set the values for the objects
            value = str(value)
            value = value.strip()  # strip the strings
            if key in nucs:  # uppercase if a nuc
                value = value.upper()
            if value:
                setattr(newBarcode, key, value)
        if not newBarcode.id_str:  # make a id_str if one is not provided
            newBarcode.id_str = str(name) + "_" + str(index)
        newBarcode.length = len(newBarcode.sequence)  # now set a default
        barCodes.append(newBarcode)  # append to our list for later saving

    # destination.close()  # now close and remove the temp file
    if index == 0:
        print(
            "Error: There must be at least one barcode! Please reload the page and try again with more barcodes.")
        return
    usedID = []
    for barCode in barCodes:
        if barCode.id_str not in usedID:
            usedID.append(barCode.id_str)
        else:
            print("Duplicate id_str for barcodes named: " + str(barCode.id_str) + ".")
            return
    usedIndex = []
    for barCode in barCodes:
        if barCode.index not in usedIndex:
            usedIndex.append(barCode.index)
        else:
            print("Duplicate index: " + barCode.index + ".")
            return
    if failed:
        print("Barcodes validation failed. The barcode set has not been saved.")
        return
    # saving to db needs to be the last thing to happen
    for barCode in barCodes:
        try:
            barCode.save()
        except:
            print("Error saving barcode to database!")
            return
    print("Barcodes Uploaded! The barcode set will be listed on the references page.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: add_barcode.py BARCODE_KIT_NAME BARCODE_CSV_PATH or  add_barcode.py BARCODE_KIT_NAME BARCODE_CSV_PATH OVERRIDE")
        sys.exit(1)

    is_override = False
    isDone = False
    if len(sys.argv) == 3:
        name, path = sys.argv[1:3]
        if command_line_add_barcode(name, path, is_override):
            isDone = True
    elif len(sys.argv) == 4:
        name, path, option = sys.argv[1:4]
        is_override = (option.lower() == "override")
        
        if command_line_add_barcode(name, path, is_override):
            isDone = True
    
    if isDone:
        print("Completed successfully")
        sys.exit(0)
    else:
        print("Failed")
        sys.exit(1)
