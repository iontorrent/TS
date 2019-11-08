import csv, codecs, cStringIO, chardet
import re


def guess_encoding_file(filename):
    # open read bytes and detect encoding and close the ZipExtFile
    with open(filename, "rU") as _tmp:
        return chardet.detect(_tmp.read())["encoding"]


def string_to_utf8(string, source_encoding):
    if source_encoding == "utf8":
        return unicode(string, "utf8")
    else:
        return unicode(string.decode(source_encoding).encode("utf8"), "utf8")


def is_ascii(text):
    """
    In Python 3.7, str, bytes, and bytearray gained support for the new isascii() method,
    which can be used to test if a string or bytes contain only the ASCII characters.
    Example code from: https://stackoverflow.com/a/35890321
    """
    if isinstance(text, unicode):
        try:
            text.encode("ascii")
        except UnicodeEncodeError:
            return False
    else:
        try:
            text.decode("ascii")
        except UnicodeDecodeError:
            return False
    return True


OnlyAscii = lambda s: re.match("^[\x00-\x7F]+$", s) != None


def utf8_to_str(utf8_string):
    return utf8_string.encode("ascii", errors="strict")


def string_list_to_utf8(string_list, source_encoding):
    return [string_to_utf8(element, source_encoding) for element in string_list]


def UnicodeDictReader(utf8_data, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield {
            unicode(key, "utf-8"): unicode(value, "utf-8") for key, value in row.items()
        }


class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """

    def __init__(self, f, encoding):
        """

        :param f: source file
        :param encoding: the encoding of source file f
        """
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")


class UnicodeReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        """
           :param f: source file
           :param encoding: the encoding of source file f
        """
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self


class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


class UnicodeDictReader:
    def __init__(
        self,
        f,
        fieldnames=None,
        restkey=None,
        restval=None,
        dialect=csv.excel,
        encoding="utf-8",
        *args,
        **kwds
    ):
        """

        :param f: the source file
        :param fieldnames:
        :param restkey:
        :param restval:
        :param dialect:
        :param encoding: the encoding of the source file
        :param args:
        :param kwds:
        """
        self._fieldnames = fieldnames  # list of keys for the dict
        self.restkey = restkey  # key to catch long rows
        self.restval = restval  # default value for short rows
        self.reader = UnicodeReader(f, dialect, encoding, **kwds)
        self.dialect = dialect
        self.line_num = 0

    def __iter__(self):
        return self

    @property
    def fieldnames(self):
        if self._fieldnames is None:
            try:
                self._fieldnames = self.reader.next()
            except StopIteration:
                pass
        self.line_num = self.reader.line_num
        return self._fieldnames

    # Issue 20004: Because DictReader is a classic class, this setter is
    # ignored.  At this point in 2.7's lifecycle, it is too late to change the
    # base class for fear of breaking working code.  If you want to change
    # fieldnames without overwriting the getter, set _fieldnames directly.
    @fieldnames.setter
    def fieldnames(self, value):
        self._fieldnames = value

    def next(self):
        if self.line_num == 0:
            # Used only for its side effect.
            self.fieldnames
        row = self.reader.next()
        self.line_num = self.reader.line_num

        # unlike the basic reader, we prefer not to return blanks,
        # because we will typically wind up with a dict full of None
        # values
        while row == []:
            row = self.reader.next()
        d = dict(zip(self.fieldnames, row))
        lf = len(self.fieldnames)
        lr = len(row)
        if lf < lr:
            d[self.restkey] = row[lf:]
        elif lf > lr:
            for key in self.fieldnames[lr:]:
                d[key] = self.restval
        return d
