# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
from iondb.rundb.ajax import render_to_json


def plupload_file_upload(request, dest_dir, request_filename_attr='name', request_files_attr='file', ):
    """ A shared file upload to save an uploaded file to disk when using plupload.
        request - the HTTPRequest object
        dest_dir - the destination directory to save the file within
    """
    if request.method == 'POST':
        name = request.POST.get(request_filename_attr, '')
        uploaded_file = request.FILES[request_files_attr]
        if not name:
            name = uploaded_file.name
        name, ext = os.path.splitext(name)

        #check to see if a user has uploaded a file before, and if they have
        #not, make them a upload directory
        if not os.path.exists(dest_dir):
            return render_to_json({"error": "upload path does not exist"})

        dest_path = '%s%s%s%s' % (dest_dir, os.sep, name, ext)

        chunk = request.POST.get('chunk', '0')
        chunks = request.POST.get('chunks', '0')

        debug = [chunk, chunks]

        with open(dest_path, ('wb' if chunk == 0 else 'ab')) as outfile:
            for content in uploaded_file.chunks():
                outfile.write(content)

        if int(chunk) + 1 >= int(chunks):
            #the upload has finished
            pass

        return render_to_json({"chuck posted": debug})

    else:
        return render_to_json({"method": "only POST here"})

def readTimezone(request):
    # read /etc/timezone into the string timezone
    timezone = ''
    with open('/etc/timezone', 'rb') as fh:
        timezone = fh.read()
        fh.close()
    return render_to_json({"timezone": timezone.strip()})
