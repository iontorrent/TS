#ifndef RGBINARY_H_
#define RGBINARY_H_

#include <stdlib.h>
#include <zlib.h>
#include "BLibDefinitions.h"

void RGBinaryRead(char*, RGBinary*, int32_t);
void RGBinaryReadBinaryHeader(RGBinary*, gzFile);
void RGBinaryReadBinary(RGBinary*, int32_t, char*);
void RGBinaryWriteBinary(RGBinary*, int32_t, char*);
void RGBinaryWriteBinaryHeader(RGBinary*, gzFile);
void RGBinaryDelete(RGBinary*);
void RGBinaryInsertBase(char*, int32_t, char);
int32_t RGBinaryGetSequence(RGBinary*, int32_t, int32_t, char, char**, int32_t);
void RGBinaryGetReference(RGBinary*, int32_t, int32_t, char, int32_t, char**, int32_t, int32_t*, int32_t*);
char RGBinaryGetBase(RGBinary*, int32_t, int32_t);
uint8_t RGBinaryGetFourBit(RGBinary*, int32_t, int32_t);
int32_t RGBinaryIsBaseRepeat(char);
//int32_t RGBinaryIsRepeat(RGBinary*, int32_t, int32_t);
#define RGBinaryIsRepeat(_rg, _contig, _position) (RGBinaryIsBaseRepeat(RGBinary_getBase(_rg, _contig, _position)))
//int32_t RGBinaryIsBaseN(char);
#define RGBinaryIsBaseN(_curBase)  ( (_curBase == 'n' || _curBase == 'N')?1:0) 
//int32_t RGBinaryIsN(RGBinary*, int32_t, int32_t);
#define RGBinaryIsN(_rg, _contig, _position)  (RGBinaryIsBaseN(RGBinaryGetBase(_rg, _contig, _positiong)))
void RGBinaryPrintInfo(char*);
void RGBinaryUnPack(RGBinary*);

#endif
