#ifndef BEVALSIM_H_
#define BEVALSIM_H_

typedef struct {
	/* Meta data */
	int readNum;
	char strand;
	int contig;
	int pos;
	int numEnds;
	int pairedEndLength;
	int readLength;
	int whichReadVariants;
	int startIndel;
	int indelLength;
	int numSNPs;
	int numErrors;
	int deletionLength;
	int insertionLength;
	/* Actual data */
	int *aContigOne;
	int *aPosOne;
	char *aStrandOne;
	int numOne;
	int *aContigTwo;
	int *aPosTwo;
	char *aStrandTwo;
	int numTwo;
} ReadType;

typedef struct {
	/* actual data */
	int numReads;
	int numAligned;
	int numCorrectlyAligned[5]; /* 0, 10, 100, 1000, 10000 */
	ReadType r;
} Stat;

typedef struct {
	Stat *stats;
	int numStats;
} Stats;

enum {OriginalRead, ReadAligned};

void ReadTypeInitialize(ReadType*);
void ReadTypeCopy(ReadType*, ReadType*);
void ReadTypePrint(ReadType*, FILE*);
int ReadTypeCompare(ReadType*, ReadType*);
int ReadTypeRead(ReadType*, gzFile, int);
void ReadTypeParseReadName(ReadType*, char*);
void ReadTypeDelete(ReadType*);

void StatsInitialize(Stats*);
void StatsPrintHeader(FILE*);
void StatsPrint(Stats*, FILE*);
void StatsAdd(Stats*, ReadType*, int);
int StatCompare(Stat*, ReadType*, int, int, char, int, int, char, int*);
void StatsDelete(Stats*);

void Evaluate(char*, char*, int);
void ReadInReads(char*, Stats*);

#endif
