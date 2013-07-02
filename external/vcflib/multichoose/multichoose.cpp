/* 

multichoose.cpp  -- n multichoose k for generic vectors

author: Erik Garrison <erik.garrison@bc.edu>
last revised: 2010-04-16

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

 */

#include <iostream>
#include <vector>
#include <list>
#include <stdlib.h>
#include "multichoose.h"


using namespace std;


int main(int argc, char** argv) { 
    if (argc < 3) {
        cerr << "usage: " << endl
             << argv[0] << " <k> <item1> <item2> ... <itemN>  ~ n multichoose k" << endl;
        return 1;
    }

    int k = atoi(argv[1]);
    vector<string> items;
    for (int i = 2; i < argc; ++i) {
        items.push_back(string(argv[i]));
    }

    vector< vector<string> > results = multichoose(k, items);

    for (vector< vector<string> >::const_iterator i = results.begin(); i != results.end(); ++i) {
        for (vector<string>::const_iterator j = i->begin(); j != i->end(); ++j) {
            cout << *j << " ";
        }
        cout << endl;
    }

    return 0; 
}
