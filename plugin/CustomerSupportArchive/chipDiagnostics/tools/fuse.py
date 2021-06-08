""" 
This module is a stripped down version of datacollect/testing/support/fuse.py
where all dependancies on c-libraries (e.g. reg) have been removed

In order to facilitate future merges, most of functions have been left intact even 
though they will not run
"""
import struct
#import reg

#max_data = (1 << reg.endpoint['fuse_wdat'].width)-1
#max_addr = (1 << reg.endpoint['fuse_addr'].width)-1
max_addr = 127
max_data = 0

TIMEOUT = 200
DEBUG = False
verbose = 0

# note: previously was 'F':'FLOWCELL
keymap = dict(L='LOT', W='WAFER', P='PART', J='JOB', C='COMMENT', B='BIN', FC='FLOWCELL')
reverse_keymap={}
for k,v in keymap.iteritems(): reverse_keymap[v]=k

def get(addr):
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    if addr > max_addr:
        raise Exception("The fuse has a max_addr = %d. Given: %d" % (max_addr, addr))
    reg.set('fuse_addr', addr)
    reg.set('fuse_read', 1)
    wait()
    return reg.get('fuse_rdat')

def set(addr, wdat):
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    if addr > max_addr:
        raise Exception("The fuse has a max_addr = %d. Given: %d" % (max_addr, addr))
    if wdat > max_data:
        raise Exception("The fuse has a max_data = %d. Given: %d" % (max_data, data))
    reg.set('fuse_addr', addr)
    reg.set('fuse_wdat', wdat)
    reg.set('fuse_write', 1)
    wait()

def wait():
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    count = 0
    while ((not reg.get('fuse_ready')) and (count < TIMEOUT)):
        count += 1
    if DEBUG: print("Waited for fuse_ready for '%d' reads." % count)
    if not (count < TIMEOUT):
        raise Exception("The fuse was not ready after %d reads." % (TIMEOUT,))

def setData(buf, addr=0x0):
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    addr_stop = addr + len(buf) - 1
    if addr_stop > max_addr:
        raise Exception("The fuse has a max_addr = %d. Required: %d" % (addr_stop, addr))
    fmt = "<%dB" % (len(buf),)
    vals = struct.unpack(fmt, buf)
    for v in vals:
        set(addr, v)
        addr = addr + 1

def getData(bytes=max_addr, addr=0x0):
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    addr_stop = addr + bytes - 1
    if addr_stop > max_addr:
        raise Exception("The fuse has a max_addr = %d. Required: %d" % (addr_stop, addr))
    x = []
    for a in range(addr, addr+bytes, 1):
        x.append(get(a))

    fmt = "<%dB" % (len(x),)
    return struct.pack(fmt, *x)

def dict_from_text(text):
    """ note, returns 'rightmost' key in efuse if multiple identical key entries
    """
    d = dict()
    l = text.split(',')
    for i in l:
        try:
            k,v = i.split(':')
            d[keymap.get(k,k)] = v
        except:
            # append current fld to 'unparsable' error key
            d['error'] = d.get('error','')+i+','
    if d.get('error'):
        if verbose:
            print 'Failed to parse efuse info: %s'%d['error']
    return d

def text_from_dict(d):
    """Convert a dictionary to text
    """
    buf = []
    for k,v in d.iteritems():
        buf.append('%s:%s'%(reverse_keymap.get(k,k),v))
    return ','.join(buf)

def getText(addr=0x0,length=0):
    """
    Read the fuse until the \x00 character is found
    """
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    x = []
    if length>0:
        for a in range(addr,min(addr+length+1,max_addr+1),1):
            val = get(a)
            x.append(val)
    else:
        for a in range(addr, max_addr+1, 1):
            val = get(a)
            if val != 0:
                x.append(val)
            else:
                break
    fmt = "<%dB" % (len(x),)
    text = struct.pack(fmt, *x)
    return text

def setDict(d):
    """Set the contents of the fuse as a dictionary
    """
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    text = text_from_dict(d)
    setText(text,EC=True)

def getDict():
    """Return the contents of the fuse as a dictionary
    """
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    text = getTextEC()
    if text == '':
        d = dict()
    else:
        try:
            d = dict_from_text(text)
            d['RAW'] = text
        except:
            print('Error parsing dict from efuse text')
            d = dict(ERROR=text)
    return d

def updateDict(d):
    """ add each key,value in d onto end of efuse as 'key:value,' 
        Consider adding check for repeat keys
    """
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    cur_buf = getTextEC()
    new_buf = text_from_dict(d)
    if cur_buf: new_buf = ','+new_buf
    # must include bad bytes in determining update start address
    setText(new_buf,EC=True,addr=len(getText()))

######################################################################
#ERROR CORRECTION PROTOCOLS

def setEC(addr, wdat, tries=1):
    """
    set a byte of data with error correction.
    return address of correctly written data.
    EC codes are 0xff and 0xfe. 0xff means that
    the block is bad. 0xfe means that the previous
    block is bad and could not be marked bad.
    """
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
#    while addr < max_addr:
#        curdat=get(addr)
#        if wdat == curdat:
#            return addr # the data is already there..
#        elif curdat == 0xff:
#	    addr += 1
#        elif get(addr+1) == 0xfe:
#            addr += 2
#        else:
#            break
#    if(curdat != 0):
#        print 'Trying to set data into non-zero space ' + addr 
    set(addr, wdat)
    failed = wdat != get(addr)
    if failed and tries > 0:
        err = 0xff #mark bad block
        set(addr, err)
        if err != get(addr): #failed to mark bad block
            addr += 1
            err = 0xfe #mark to indicate previous block is bad
            set(addr, err)
            if err != get(addr): #failed to mark previous error
                print "Failed to mark error in fuse..."
            else:
                print "addr = %d...marked bad block(s) correctly in fuse..." % addr
        else:
            print "addr = %d...marked bad block correctly in fuse..." % addr
        return setEC(addr+1, wdat, tries=tries-1)
    else:
        if failed:
            print "addr = %d...wrote bad data in fuse..." % addr
        return addr

def setText(buf, EC=True, addr=0x0):
    """Write text buffer to the fuse.
    If the fuse fails to write a byte, it should mark the
    entire byte as 0xff. If the 0xff, fails it should write
    the next byte as 0xfe.
    """
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    retval = True
    addr_stop = addr + len(buf) - 1
    if addr_stop > max_addr:
        raise Exception("The fuse has a max_addr = %d. Required: %d" % (addr_stop, addr))
    fmt = "<%dB" % (len(buf),)
    vals = struct.unpack(fmt, buf)
    for v in vals:
        if EC:
            addr = setEC(addr, v)
        else:  # TODO: set doesn't seem to work, always use setEC's for now
            #addr = set(addr,v)
            set(addr,v)
        addr += 1

def setTextEC(buf, addr=0x0):
    """Write text buffer to the fuse.
    If the fuse fails to write a byte, it should mark the
    entire byte as 0xff. If the 0xff, fails it should write
    the next byte as 0xfe.
    """
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    retval = True
    addr_stop = addr + len(buf) - 1
    if addr_stop > max_addr:
        raise Exception("The fuse has a max_addr = %d. Required: %d" % (addr_stop, addr))
    fmt = "<%dB" % (len(buf),)
    vals = struct.unpack(fmt, buf)
    for v in vals:
        addr = setEC(addr, v)
        addr += 1

def getTextEC(addr=0x0):
    """
    Read the fuse until the \x00 character is found and
    correct errors by the error codes
    """
    raise NotImplementedError( 'This function depends on C libraries.  Use the version at /software/testing/support instead' )
    x = []
    for a in range(addr, max_addr+1, 1):
        val = get(a)
        if val != 0:
            x.append(val)
        else:
            break

    #Error correction
    xn = []
    addr_end = len(x)-1
    i = 0
    while i < len(x):
        if x[i] == 0xff:
            if DEBUG: print "addr = %d...identified bad block correctly in fuse..." % (addr+i,)
            i += 1 #bad block marked, skip it
        elif i != addr_end and x[i+1] == 0xfe:
            if DEBUG: print "addr = %d...identified bad block(s) correctly in fuse..." % (addr+i,)
            i += 2 #corrupt block marked by next block
        else:
            xn.append(x[i])
            i += 1
                
    fmt = "<%dB" % (len(xn),)
    text = struct.pack(fmt, *xn)
    return text

    
