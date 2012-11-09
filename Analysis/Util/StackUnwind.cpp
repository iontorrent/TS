/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#define UNW_LOCAL_ONLY
#include <libunwind.h>

using namespace std;

static ofstream dumpStr;

static int PrintStack(ostream& out)
{
  // Unwind the stack, printing function names along the way.
  unw_cursor_t  cursor;
  unw_context_t context;

  int err = unw_getcontext(&context);
  if(err) return err;

  err = unw_init_local(&cursor, &context);
  if(err) return err;

  while ((err = unw_step(&cursor)) > 0) {
    unw_word_t pc = 0;
    err = unw_get_reg(&cursor, UNW_REG_IP, &pc);
	if(err) break;

    unw_word_t offset = 0;
    char fname[1024];
    fname[0] = '\0';
    err = unw_get_proc_name(&cursor, fname, sizeof(fname), &offset);
	if(err) break;

    out << "0x" << setw(18) << left << hex << pc
        << fname
        << " + 0x" << offset
        << dec
        << endl;
  }

  return err;
}

static void HandleSig(ostream& out, int sig)
{
  // Print the stack.
  out << "received signal " << sig << ": " << strsignal(sig) << endl;
  int err = PrintStack(out);
  if(err)
  	out << "problem crawling stack: " << err << endl;
}

static void HandleSys(int sig)
{
  // Print the stack to a file, and die.
  HandleSig(dumpStr, sig);
  abort();
}

static void HandleUsr(int sig)
{
  // Print the stack to standard out:
  HandleSig(cout, sig);
}

void InitStackUnwind(const char* dumpFileName)
{
  if(dumpFileName){
    dumpStr.open(dumpFileName);
    assert(dumpStr);

    signal(SIGSEGV, HandleSys);
    signal(SIGQUIT, HandleSys);
    signal(SIGILL,  HandleSys);
    signal(SIGFPE,  HandleSys);
    signal(SIGSYS,  HandleSys);
    signal(SIGBUS,  HandleSys);
  }

  signal(SIGUSR1, HandleUsr);
}

