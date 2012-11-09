/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <iomanip>
#include <sys/types.h>
#include <unistd.h>
#include "StackUnwind.h"

// Test code for StackUnwind.cpp.

using namespace std;

void crash()
{
	int* p = 0;
	cout << *p;
}

void f3() {cerr << "f3" << endl; crash();}
void f2() {cerr << "f2" << endl; f3();}
void f1() {cerr << "f1" << endl; f2();}
void f0() {cerr << "f0" << endl; f1();}

int main(int argc, char* argv[])
{
  InitStackUnwind();

  if(argc == 1)
    f0();

  cout << getpid() << endl;
  int n = 0;
  while(1){
	cout << n++ << endl;
	sleep(1);
  }
}


