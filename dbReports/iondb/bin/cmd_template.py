# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import csv
import json
from optparse import OptionParser
from HTMLParser import HTMLParser


# For easy parsing of html content.May or may not use
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        print(attrs)

    def handle_endtag(self, tag):
        print(tag)

    def handle_data(self, data):
        print(data)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="inputFile", help="html filename to parse")
    parser.add_option(
        "-o",
        "--output",
        dest="outputFile",
        help="output file to direct the json generated from stripping div tag",
    )
    parser.add_option(
        "--json",
        dest="jsonFile",
        help="If you have more than one divs, enter the json filename",
    )
    (options, args) = parser.parse_args()

    dictionary = {}
    with open(str(options.jsonFile), "r") as f:
        out = f.readlines()
        out = "".join(out)
        dictionary = json.loads(out)

    divArray = list(dictionary.keys())

    # File operations, Open and read on file
    fileIn = open(str(options.inputFile), "r")
    mainArray = []
    for row in fileIn:
        array = []
        userJson = {}
        if "<div" in row:
            for x in divArray:
                if x in row:
                    row1 = row
                    for row1 in fileIn:
                        if "</div>" in row1:
                            userJson["content"] = "".join(array)
                            userJson["atitle"] = dictionary[x]["title"]
                            userJson["caption"] = dictionary[x]["caption"]
                            mainArray.append(userJson)
                            break
                        else:
                            array.append(row1.rstrip("\n"))
    fileIn.close()
    # Create a json structure

    # dump the dictionary into json and write out to a file
    p = json.dumps(mainArray)
    fileOut = open(str(options.outputFile), "w")
    fileOut.write(p)
    fileOut.close()
