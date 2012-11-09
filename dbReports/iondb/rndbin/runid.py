#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
def create_runid(name):
    
    def DEKHash(key):
        hash = len(key)
        for i in key:
            hash = ((hash << 5) ^ (hash >> 27)) ^ ord(i)
        return (hash & 0x7FFFFFFF)

    def base10to36(num):
        str = ''
        for i in range(5):
            digit=num % 36
            if digit < 26:
                str = chr(ord('A') + digit) + str
            else:
                str = chr(ord('0') + digit - 26) + str
            num /= 36
        return str
    
    print "Output from DEKHash: %s" % DEKHash(name)
    print "Output from DEKHash: %s" % base10to36(DEKHash(name))
    
    return 

if __name__ == '__main__':
    create_runid("Auto_BLACKBIsadRD_EXPLdOffg_205asdfklajsdhfklajsdhf_490")