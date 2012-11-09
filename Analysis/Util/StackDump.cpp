/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <execinfo.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include "StackDump.h"

using namespace std;

static const size_t bufSize = 2048;
static char pidBuf[32];
static char nameBuf[bufSize];

static void PrintStack(int sig)
{
  cout << nameBuf << endl;
  cout << "pid " << pidBuf << endl;
  cout << "received signal " << sig << endl;

  // Fork a child to run gdb:
  int child_pid = fork();
  if (child_pid == 0) {
    // We are the child; exec gdb:
    execlp("gdb", "gdb", "--batch", "-n", "-ex", "thread", "-ex", "bt full", nameBuf, pidBuf, (char*)0);
	cout << "failed to exec gdb: " << errno << endl;
    abort(); // if gdb failed to start
  } else if(child_pid > 0) {
    // We are the parent, wait for gdb to wrap:
    waitpid(child_pid, 0, 0);
  } else {
    // Uh oh:
    cout << "couldn't fork: " << errno << endl;
  }

  // That's all, folks!
  exit(1);
}

void InitStackDump()
{
  // Get our pid:
  sprintf(pidBuf, "%d", getpid());

  // And the name of our process:
  size_t       read_len = readlink("/proc/self/exe", nameBuf, bufSize-1);
  assert(0 < read_len and read_len < bufSize);

  // Install handlers:
  signal(SIGSEGV, ::PrintStack);
  signal(SIGQUIT, ::PrintStack);
  signal(SIGILL,  ::PrintStack);
  signal(SIGABRT, ::PrintStack);
  signal(SIGFPE,  ::PrintStack);
  signal(SIGSYS,  ::PrintStack);
  signal(SIGBUS,  ::PrintStack);
}

