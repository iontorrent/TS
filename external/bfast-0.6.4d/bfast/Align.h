#ifndef ALIGN_H_
#define ALIGN_H_
#include "BLibDefinitions.h"
#include "AlignMatrix.h"

/* For the "from" for NT data */
enum {StartNT, /* 0 */
	DeletionStart, /* 1 */
	DeletionExtension, /* 2 */
	Match, /* 3 */
	InsertionStart, /* 4 */
	InsertionExtension, /* 5 */
	NoFromNT}; /* 6 */

/* For the "from" for CS data */
enum {
	StartCS, /* 0 */
	DeletionA, /* 1 */
	DeletionC, /* 2 */
	DeletionG, /* 3 */
	DeletionT, /* 4 */
	DeletionN, /* 5 */
	MatchA, /* 6 */
	MatchC, /* 7 */
	MatchG, /* 8 */
	MatchT, /* 9 */
	MatchN, /* 10 */
	InsertionA, /* 11 */
	InsertionC, /* 12 */
	InsertionG, /* 13 */
	InsertionT, /* 14 */
	InsertionN, /* 15 */
	NoFromCS /* 16 */
};

int AlignRGMatches(RGMatches*, RGBinary*, AlignedRead*, int32_t, int32_t, ScoringMatrix*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, AlignMatrix*);
void AlignRGMatchesOneEnd(RGMatch*, RGBinary*, AlignedEnd*, int32_t, int32_t, ScoringMatrix*, int32_t, int32_t, int32_t, double*, int32_t*, AlignMatrix*);
int32_t AlignExact(char*, int32_t, char*, int32_t, ScoringMatrix*, AlignedEntry*, int32_t, int32_t, int32_t, char);
int32_t AlignUngapped(char*, char*, char*, int32_t, char*, int32_t, int32_t, ScoringMatrix*, AlignedEntry*, int32_t, int32_t, int32_t, char);
void AlignGapped(char*, char*, char*, int32_t, char*, int32_t, int32_t, ScoringMatrix*, AlignedEntry*, AlignMatrix*, int32_t, int32_t, int32_t, int32_t, int32_t, char, double);
void AlignGappedBounded(char*, char*, int32_t, char*, int32_t, ScoringMatrix*, AlignedEntry*, AlignMatrix*, int32_t, int32_t, char, double, int32_t, int32_t);
void AlignGappedConstrained(char*, char*, char*, int32_t, char*, int32_t, ScoringMatrix*, AlignedEntry*, AlignMatrix*, int32_t, int32_t, int32_t, int32_t, int32_t, char);
int32_t AlignRGMatchesKeepBestScore(AlignedEnd*, double);

#endif
