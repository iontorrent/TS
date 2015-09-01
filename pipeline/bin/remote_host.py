#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
"""Attempt to open a socket connection to each of a list of HOST:PORT pairs
"""
__author__ = 'bakennedy'


import socket
import threading
import optparse
import sys
from Queue import Queue


def check_connection(connection_string, timeout=10):
    try:
        host, port = connection_string.split(':')
        port = int(port)
    except Exception as err:
        print('The connection string "%s" is invalid: %s' %
              (connection_string, err))
        return False
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
    except Exception as err:
        return False
    else:
        return True


def print_result(connection_string, result):
    format = "%s\t%s"
    fail = "\033[91mFAILED\033[0m"
    success = "\033[92mOK\033[0m"
    print(format % ((success if result else fail), connection_string))


def queued_check(queue, connection_string, timeout=10):
    result = check_connection(connection_string, timeout)
    queue.put((connection_string, result))


ion_remotes = [
    "rssh.iontorrent.com:22",
    "drm.appliedbiosystems.com:443",
    "ionupdates.com:80",
    "us.archive.ubuntu.com:80",
    "security.ubuntu.com:80"
]


if __name__ == "__main__":
    parser = optparse.OptionParser(usage="%prog [options] [HOST:PORT]...",
                                   description=__doc__)
    parser.add_option("-t", "--timeout", type=float, default=10,
        help="Timeout each connection after TIMEOUT seconds (default 10)")
    parser.add_option("-a", "--asap", action="store_true",
        help="Output the results as soon as they arrive rather than in order (Useful if you know the timeout should be long but get bored easily)")
    parser.add_option("-l", "--list", action="store_true",
        help="Print a list of the default remote hosts that would be checked and exit.")
    parser.add_option("-c", "--clean", action="store_true", help="Exit status 0 whether or not any connections fail.  By default the exit status is 1 if any connections fail.")
    (opts, args) = parser.parse_args()
    if opts.list:
        print("\n".join(ion_remotes))
        sys.exit()
    remotes = args or ion_remotes

    queue = Queue()
    threads = []
    for remote in remotes:
        t = threading.Thread(target=queued_check,
                             args=(queue, remote, opts.timeout))
        threads.append(t)
        t.start()

    if opts.asap:
        results = dict()
        while len(results) < len(remotes):
            remote, result = queue.get()
            results[remote] = result
            print_result(remote, result)

    for thread in threads:
        thread.join()

    if not opts.asap:
        results = dict(queue.queue)
        for remote in remotes:
            print_result(remote, results[remote])

    sys.exit(0 if all(results.values()) else 1)
