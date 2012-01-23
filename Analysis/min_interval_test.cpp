/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert>
#include <iostream>
#include <iomanip>
#include <vector>
#include "min_interval.h"

using namespace std;

int main()
{
	int x[] = {1,0,1,1,1,1,3};
	size_t len = sizeof(x) / sizeof(x[0]);
	vector<int> y(x,x+len);

	pair<vector<int>::iterator,vector<int>::iterator> mi0 = min_interval(y.begin(), y.end(), 5);
	cout << setw(4) << mi0.first-y.begin() << setw(4) << mi0.second-y.begin() << endl;
	
	pair<int*,int*> mi1 = min_interval(x, x+len, 5);
	cout << setw(4) << mi1.first-x << setw(4) << mi1.second-x << endl;
}


