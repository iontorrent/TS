/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TFS_H
#define TFS_H

#include "LinuxCompat.h"

struct TFInfo {
        char    name[64];
        char    key[64];
        char    seq[1024];       // sequence for this TF
        int     len;            // length of this TF
        int     count;          // how many of these we find
	int	Ionogram[800];	// expected ionogram signal per flow
	int	flows;		// num of flows in the Ionogram
};

class TFs {
	public:
		TFs(const char *flowOrder);
		virtual ~TFs();
		bool	LoadConfig(char *file); // returns true if successful, false otherwise and sets tfInfo to the default set of TF's
		TFInfo	*Info() {return tfInfo;}
		int	Num() {return numTFs;}
		char	*GetTFConfigFile ();
		int	GenerateIonogram(const char *seq, int len, int *ionogram);
	protected:
		TFInfo	*tfInfo;
		int	numTFs;
		void	UpdateIonograms();
		char	*flowOrder;
		int	numFlowsPerCycle;
};

class TFTracker {
	public:
		TFTracker (char *experimentName);
		virtual ~TFTracker();
		bool 	Close();
		bool 	Add (int row, int col, char *label);
		
	private:
		char	*fileName;
		FILE	*fd;
};

#endif // TFS_H

