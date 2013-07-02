# encoding: utf-8

"""

multichoose.py  -- non-recursive n multichoose k for python lists

author: Erik Garrison <erik.garrison@bc.edu>
last revised: 2010-04-30

Copyright (c) 2010 by Erik Garrison

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""

def multichoose(k, objects):
    """n multichoose k multisets from the list of objects.  n is the size of
    the objects."""
    j,j_1,q = k,k,k  # init here for scoping
    r = len(objects) - 1
    a = [0 for i in range(k)] # initial multiset indexes
    while True:
        yield [objects[a[i]] for i in range(0,k)]  # emit result
        j = k - 1
        while j >= 0 and a[j] == r: j -= 1
        if j < 0: break  # check for end condition
        j_1 = j
        while j_1 <= k - 1:
            a[j_1] = a[j_1] + 1 # increment
            q = j_1
            while q < k - 1:
                a[q+1] = a[q] # shift left
                q += 1
            q += 1
            j_1 = q

