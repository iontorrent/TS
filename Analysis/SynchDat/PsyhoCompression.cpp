/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#define INLINE   __attribute__ ((always_inline))
#define NOINLINE __attribute__ ((noinline))

#define ALIGNED __attribute__ ((aligned(16)))

#define likely(x)   __builtin_expect(!!(x),1)
#define unlikely(x) __builtin_expect(!!(x),0)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>
#include <map>
#include <set>
#include <queue>
#include <x86intrin.h>
#include "PsyhoCompression.h"

using namespace std;

#define FOR(i,a,b)  for(int i=(a);i<(int)(b);++i)
#define FORS(i,a,b,c)  for(int i=(a);i<(int)(b);i+=c)
#define REP(i,a)    FOR(i,0,a)
#define ZERO(m)     memset(m,0,sizeof(m))
#define ALL(x)      x.begin(),x.end()
#define PB          push_back
#define S           size()
#define LL          long long
#define LD          long double
#define MP          make_pair
#define X           first
#define Y           second
#define VC          vector
#define PII         pair <int, int>
#define VI          VC < int >
#define VPII        VC < PII >
#define VVI         VC < VI >
#define VVVI        VC < VVI >
#define VD          VC < double >
#define VS          VC < string >
#define DB(a)       cout << #a << ": " << (a) << endl;


typedef signed char INT8;
typedef unsigned char UINT8;
typedef UINT8* PUINT8;
typedef signed short INT16;
typedef INT16* PINT16;
typedef unsigned short UINT16;
typedef UINT16* PUINT16;
typedef unsigned int UINT32;
typedef UINT32* PUINT32;
typedef signed int INT32;
typedef INT32* PINT32;
typedef unsigned long long UINT64;

void print(VI v) {cout << "[";if (v.S) cout << v[0];FOR(i, 1, v.S) cout << ", " << v[i];cout << "]\n";}
void print(VC < LL > v) {cout << "[";if (v.S) cout << v[0];FOR(i, 1, v.S) cout << ", " << v[i];cout << "]\n";}
void print(VD v) {cout << "[";if (v.S) cout << v[0];FOR(i, 1, v.S) cout << ", " << v[i];cout << "]\n";}
void print(VS v) {cout << "[";if (v.S) cout << v[0];FOR(i, 1, v.S) cout << ", " << v[i];cout << "]\n";}
template<class T> string i2s(T x) {ostringstream o; o << x; return o.str(); }
VS splt(string s, char c = ' ') {VS rv; int p = 0, np; while (np = s.find(c, p), np >= 0) {if (np != p) rv.PB(s.substr(p, np - p)); p = np + 1;} if (p < (int)s.S) rv.PB(s.substr(p)); return rv;}

#define REDUCE 0.01

#define MAX_VAL ((1<<14)-1)
#define MAX_SL 64

#define MAX_TABLE (MAX_SL * 16 + 1)
#define VF_TABLE (MAX_TABLE - 1)

#define MAX_DIFF ((1<<14) / SCALE + 2)

#define SUM_STEP 45
#define SHIFT_SIZE 200

#define MIN_RANGE (1 << 16)
#define MAX_COUNT_BITS 16
#define MAX_COUNT (1 << MAX_COUNT_BITS)

struct RangeEncoder {
    const PUINT8 COMP;
    UINT32 POS;
    
    UINT32 range;
    UINT32 low;
    
    PUINT32 tdata[MAX_TABLE];
    PUINT32 tptr[MAX_TABLE];
    
    RangeEncoder(PUINT8 ptr) : COMP(ptr), POS(0), range(0xFFFFFFFFU), low(0) { 
        REP(i, MAX_TABLE) tdata[i] = NULL;
    }
    
    ~RangeEncoder() {
        REP(i, MAX_TABLE) if (tdata[i]) delete[] tdata[i];
    }
    
    INLINE void flush() {
        REP(i, 4) {
            COMP[POS++] = low >> 24;
            low <<= 8;
        }
    }
    
    INLINE void normalize() {
      while (((low ^ low) + range) < 0x01000000U || (range < MIN_RANGE && (range = -low))) {
            COMP[POS++] = low >> 24;
            range <<= 8;
            low <<= 8;
        }
        
    }
    
    INLINE void encode(UINT32 table, UINT32 symbol) {
        range >>= MAX_COUNT_BITS;
        low += tptr[table][symbol] * range;
        range *= tptr[table][symbol + 1] - tptr[table][symbol];
        normalize();
    }
    
    template < int SIZE > INLINE void encode(UINT32 x) {
        range >>= SIZE;
        low += x * range;
        normalize();
    }
    
    void encodeTable(UINT32 table, VI &counts) {
        int lo = -1, hi = -1, mx = -1;
        int sum = 0;
        REP(i, counts.S) if (counts[i]) {
            if (lo == -1) lo = i, mx = i;
            hi = i;
            if (counts[i] > counts[mx]) mx = i;
            sum += counts[i];
        }
        
        if (sum == 0) {
            encode<1>(0);
            return;
        }
        
        encode<1>(1);
        
        tdata[table] = new UINT32[hi - lo + 2];
        tptr[table] = tdata[table] - lo;
        
        tptr[table][hi + 1] = 0;
        
        double ratio = (double)MAX_COUNT / sum;
        
	again:
        INT32 total = MAX_COUNT;
        FOR(i, lo, hi + 1) if (i != (int)mx) {
            tptr[table][i] = counts[i] ? max(1U, (UINT32)(counts[i] * ratio)) : 0;
            total -= tptr[table][i];
        }
        
        if (total <= 0.8 * counts[mx] * ratio) {
            ratio *= 0.99;
            goto again;
        }
        tptr[table][mx] = total;
        
        for (int i = hi + 1; i > lo; i--)
            tptr[table][i] = tptr[table][i - 1];
        tptr[table][lo] = 0;
		
        FOR(i, lo, hi + 1)
		tptr[table][i + 1] += tptr[table][i];
        
        encode<12>(lo);
        encode<12>(hi);
        FOR(i, lo + 1, hi + 1) encode<MAX_COUNT_BITS>(tptr[table][i]);
        
    }
    
};

struct RangeDecoder {
    const PUINT8 COMP;
    UINT32 POS;
    
    UINT32 range;
    UINT32 low;
    UINT32 data;
    
    PUINT32 tdata[MAX_TABLE];
    PUINT32 pdata[MAX_TABLE];
    
    UINT32 tlo[MAX_TABLE];
    
    RangeDecoder(PUINT8 ptr) : COMP(ptr), POS(0), range(0xFFFFFFFFU), low(0), data(0) {
        REP(i, 4) data = (data << 8) | COMP[POS++];
        REP(i, MAX_TABLE) tdata[i] = NULL;
    }
    
    ~RangeDecoder() {
        REP(i, MAX_TABLE) if (tdata[i]) delete[] tdata[i];
    }
    
    INLINE void normalize() {
      while (((low ^ low) + range) < 0x01000000U || (range < MIN_RANGE && (range = -low))) {
            data = (data << 8) | COMP[POS++];
            range <<= 8;
            low <<= 8;
        }
    }
    
    INLINE UINT32 decode(UINT32 table) {
        range >>= MAX_COUNT_BITS;
        UINT32 pos = (data - low) / range;
        
        /*
		 int tpos = 1;
		 for (; pos >= tdata[table][tpos]; tpos++);
		 UINT32 rv = tpos - 1 + tlo[table];
		 */
        int tpos;
        int i = 0;
        while (true) {
            tpos = pdata[table][i];
            if (pos >= tdata[table][tpos] && pos < tdata[table][tpos + 1]) break;
            i++;
        }
        UINT32 rv = tpos + tlo[table];
        
        low += tdata[table][tpos] * range;
        range *= tdata[table][tpos + 1] - tdata[table][tpos];
        normalize();
        return rv;
    }
    
    template < int SIZE > INLINE UINT32 decode() {
        range >>= SIZE;
        UINT32 rv = (data - low) / range;
        low += rv * range;
        //range *= 1;
        normalize();
        return rv;
    }
    
    void decodeTable(UINT32 table) {
        int exist = decode<1>();
        //cout << table << ' ' << exist << endl;
        if (!exist) return;
		
        UINT32 lo = decode<12>();
        UINT32 hi = decode<12>();
        
        tdata[table] = new UINT32[hi - lo + 2];
        pdata[table] = new UINT32[hi - lo + 2];
        
        tdata[table][0] = 0;
        FOR(i, 1, hi - lo + 1)
		tdata[table][i] = decode<MAX_COUNT_BITS>();
        tdata[table][hi - lo + 1] = MAX_COUNT;
        
        VPII vp(hi - lo + 1);
        REP(i, hi - lo + 1)
		vp.PB(MP(tdata[table][i+1] - tdata[table][i], i));
        sort(vp.rbegin(), vp.rend());
        REP(i, hi - lo + 1)
		pdata[table][i] = vp[i].Y;
        
        tlo[table] = lo;
    }
    
    
};

/*
 template <int DIST, int SHIFT> INLINE void SIMD_DIFF(PINT16 TV, PINT16 VAL) {
 for (int l = 0; l < 48; l += 8) {
 __m128i mtv0 = _mm_load_si128((__m128i*)&TV[l]);
 __m128i mtv1 = _mm_loadu_si128((__m128i*)&TV[l+DIST]);
 __m128i m = _mm_sub_epi16(mtv0, mtv1);
 if (SHIFT) m = _mm_slli_epi16(m, SHIFT);
 
 __m128i mval0 = _mm_load_si128((__m128i*)&VAL[l]);
 mval0 = _mm_sub_epi16(mval0, m);
 _mm_store_si128((__m128i*)&VAL[l], mval0);
 
 __m128i mval1 = _mm_loadu_si128((__m128i*)&VAL[l+DIST]);
 mval1 = _mm_add_epi16(mval1, m);
 _mm_storeu_si128((__m128i*)&VAL[l+DIST], mval1);
 }
 }
 
 template <int DIST, int SHIFT, class T> INLINE void CALC_DIFF(T *TV, T *VAL, int sl) {
 int last = sl - DIST - 1;
 for (int l = 0; l < last; l += 2) {
 int d0;
 if (SHIFT)
 d0 = (TV[l] - TV[l+DIST]) << SHIFT;
 else
 d0 = (TV[l] - TV[l+DIST]);
 VAL[l] -= d0;
 VAL[l+DIST] += d0;
 
 int d1;
 if (SHIFT)
 d1 = (TV[l+1] - TV[l+1+DIST]) << SHIFT;
 else
 d1 = (TV[l+1] - TV[l+1+DIST]);
 VAL[l+1] -= d1;
 VAL[l+1+DIST] += d1;
 }
 if ((last & 1) == 0) {
 int d;
 if (SHIFT) 
 d = (TV[last] - TV[last + DIST]) << SHIFT;
 else
 d = TV[last] - TV[last + DIST];
 VAL[last] -= d;
 VAL[last + DIST] += d;
 }
 }
 */

NOINLINE double testDiff(PUINT16 data, VVVI &sum, int sx, int sy, int sl, int scale, int step, int TYPE = 0, int *TC = NULL) {
    VVVI COUNT(MAX_SL, VVI(SHIFT_SIZE, VI(scale, 0)));
    int steps = 0;
    const int MD = scale;
    const int MD2 = TYPE ? MD * 1 / 3 : MD * 1 / 2;
    const int MD3 = TYPE ? MD * 6 / 5 : MD * 4 / 3;
    const int MD4 = MD - 4;
	
    
    LL sumdiff = 0;
    
    for (int x = 0; x < sx; x += step) for (int y = 0; y < sy; y += step) {    
        PUINT16 p = &data[(x * sy + y) * sl];
        if (p[0] == MAX_VAL + 1) continue;
        
        INT32 TV[MAX_SL];   
        int avg = (p[0] + scale * 2) / (scale * 4) * (scale * 4);
        VI &v = sum[x / SUM_STEP][y / SUM_STEP];
        REP(l, sl) TV[l] = p[l] - v[l] - avg;
        
        INT32 COFF[MAX_SL];
        int off = 0;
        
        REP(l, sl) {
            int d = TV[l] + off * MD;
            int dd = l < (int)sl - 1 ? TV[l+1] + off * MD : 0;
            
            int x = 0;
            if (((d <= -MD || d <= -MD4) && dd <= -MD4) && !(dd >= -MD2 && d >= -MD3)) {
                x = max(1, -d / MD);
            } else if ((d >= MD || (d >= MD4 && dd >= MD4)) && !(dd <= MD2 && d <= MD3)) {
                x = -max(1, d / MD);
            }
            
            off += x;
            
            COFF[l] = off;
        }
        
        for (int l = sl - 2; l >= 0; l--) {
            if (COFF[l] == COFF[l+1]) continue;
            int d = TV[l] + COFF[l] * MD;
            if (COFF[l] < COFF[l+1] && d < -MD / 2) 
                COFF[l]++;
            else if (COFF[l] > COFF[l+1] && d > MD / 2)
                COFF[l]--;
            else if (l && COFF[l] < COFF[l+1] && COFF[l] < COFF[l-1] && d > -MD)
                COFF[l]++;
            else if (l && COFF[l] > COFF[l+1] && COFF[l] > COFF[l-1] && d < MD)
                COFF[l]--;
        }
        
        REP(l, sl) {
            int d = TV[l] + COFF[l] * MD;
            sumdiff += d * d;
        }
        
        steps++;
    }
    
    /*
	 LL sum = 0;
	 REP(l, MAX_SL) {
	 int lbr = 0;
	 REP(i, SHIFT_SIZE) {
	 LL bv = 1ULL << 60;
	 int br = -1;
	 FOR(j, max(0, lbr - 10), scale) {
	 //REP(j, 50) {
	 LL av = 0;
	 REP(k, scale) {
	 int d = (k - j) * (k - j);
	 av += (LL)COUNT[l][i][k] * d;
	 }
	 if (av < bv) {
	 bv = av;
	 br = j;
	 } 
	 else break;
	 }
	 sum += bv;
	 lbr = br;
	 if (TC) TC[l * SHIFT_SIZE + i] = br;
	 }
	 }
	 
	 if (TC) {
	 REP(l, 60) {
	 int sv;
	 sv = TC[l * SHIFT_SIZE + SHIFT_SIZE / 2];
	 for (int i = SHIFT_SIZE / 2 + 1; i < SHIFT_SIZE; i++) {
	 if (TC[l * SHIFT_SIZE + i]) sv = TC[l * SHIFT_SIZE + i];
	 else TC[l * SHIFT_SIZE + i] = sv;
	 }
	 sv = TC[l * SHIFT_SIZE + SHIFT_SIZE / 2];
	 for (int i = SHIFT_SIZE / 2 - 1; i >= 0 ; i--) {
	 if (TC[l * SHIFT_SIZE + i]) sv = TC[l * SHIFT_SIZE + i];
	 else TC[l * SHIFT_SIZE + i] = sv;
	 }
	 }
	 }
	 */
    
    return sumdiff / 1.0 / steps / sl;
}

int PsyhoDatCompression::GetCompressionType() { return 2; }
	
VI PsyhoDatCompression::compress(const VI &datavi) {
  PUINT16 data = (PUINT16)&datavi[0];
		
  int sx = data[0];
  int sy = data[1];
  int sl = data[2];
		
  VI rv;
  rv.reserve(datavi.S);
  ((int**)&rv)[1] = ((int**)&rv)[0] + rv.capacity();
		
  RangeEncoder re((PUINT8)&rv[0]);
		
  re.encode<10>(sx);
  re.encode<10>(sy);
  re.encode<10>(sl);
		
  int p = 3;
		
  VVVI sum(sx / SUM_STEP + 1, VVI(sy / SUM_STEP + 1, VI(sl, 0)));
  VVI sumno(sx / SUM_STEP + 1, VI(sy / SUM_STEP + 1, 0));
  //  int sameno = 0;
  REP(x, sx) REP(y, sy) {
    bool same = data[p] == 0 || data[p] == MAX_VAL;
    FOR(l, 1, sl) if (data[p] != data[p+l]) {
      same = false;
      break;
    }
			
    if (same) {
      re.encode<1>(data[p] == MAX_VAL);
      re.encode<10>(x);
      re.encode<10>(y);
      data[p] = MAX_VAL + 1;
      p += sl;
      continue;
    }
			
    int px = x / SUM_STEP;
    int py = y / SUM_STEP;
    sumno[px][py]++;
    REP(l, sl)
      sum[px][py][l] += data[p++];
  }
		
  re.encode<1>(0);
  re.encode<10>(sx);
  re.encode<10>(sy);
		
  VI CSUM(1 << 14, 0);
  REP(x, sum.S) REP(y, sum[x].S) {
    if (sumno[x][y] == 0) continue;
    int no = sumno[x][y];
    REP(l, sl) sum[x][y][l] = (sum[x][y][l] + no / 2) / no;
    int zero = sum[x][y][0];
    REP(l, sl) sum[x][y][l] -= zero;
    REP(l, sl) CSUM[(1<<13) + sum[x][y][l]]++;
  }
		
  int SCALE = 10;
		
  while (true) {
    double est = testDiff(&data[3], sum, sx, sy, sl, SCALE, 10, 0, NULL);
    if (est > 37.0) break;
    SCALE++;
  }
  SCALE--;
		
  int TYPE = 0;
		
  int TC[MAX_SL * SHIFT_SIZE];    
  while (true) {
    double est = testDiff(&data[3], sum, sx, sy, sl, SCALE, 4, TYPE, TC);
    DB(est);
    if (est < 35.6)
      break;
    if (TYPE == 0) {
      TYPE = 1;
    } else {
      TYPE = 0;
      SCALE--;
    }
  }
  DB(SCALE);
		
  const int MD = SCALE;
  const int MD2 = TYPE ? MD * 1 / 3 : MD * 1 / 2;
  const int MD3 = TYPE ? MD * 6 / 5 : MD * 4 / 3;
  const int MD4 = MD - 4;
		
  re.encode<8>(SCALE);    
		
  REP(x, sum.S) REP(y, sum[x].S) FOR(l, 1, sl) {
    int d = sum[x][y][l] - sum[x][y][l-1];
    if (abs(d) < 30) {
      re.encode<1>(0);
      re.encode<6>(32 + d);
    } else {
      re.encode<1>(1);
      re.encode<15>((1<<14) + d);
    }
  }
		
  //REP(i, MAX_SL * SHIFT_SIZE) re.encode<8>(TC[i]);
		
  VVI COUNT(VF_TABLE + 1, VI(MAX_DIFF * 2, 0));
		
  p = 3;
		
  VI AVG(sx * sy);
  REP(x, sx) REP(y, sy) {
    if (data[p] == MAX_VAL + 1) {
      p += sl;
      continue;
    }
            
    INT32 TV[MAX_SL];   
    int avg = (data[p] + SCALE * 2) / (SCALE * 4) * (SCALE * 4);
    AVG[x * sy + y] = avg;
    COUNT[VF_TABLE][avg / (SCALE * 4)]++;
			
    VI &vs = sum[x / SUM_STEP][y / SUM_STEP];
    REP(l, sl) TV[l] = data[p + l] - vs[l] - avg;
			
    INT32 COFF[MAX_SL];
    int off = 0;
			
    REP(l, sl) {
      int d = TV[l] + off * MD;
      int dd = l < (int)(sl - 1) ? TV[l+1] + off * MD: 0;
				
      int x = 0;
      if ((d <= -MD || (d <= -MD4 && dd <= -MD4)) && !(dd >= -MD2 && d >= -MD3)) {
        x = max(1, -d / MD);
      } else if ((d >= MD || (d >= MD4 && dd >= MD4)) && !(dd <= MD2 && d <= MD3)) {
        x = -max(1, d / MD);
      }
				
      off += x;
				
      COFF[l] = off;
    }
			
    for (int l = sl - 2; l >= 0; l--) {
      if (COFF[l] == COFF[l+1]) continue;
      int d = TV[l] + COFF[l] * MD;
      if (COFF[l] < COFF[l+1] && d < -MD / 2) 
        COFF[l]++;
      else if (COFF[l] > COFF[l+1] && d > MD / 2)
        COFF[l]--;
      else if (l && COFF[l] < COFF[l+1] && COFF[l] < COFF[l-1] && d > -MD)
        COFF[l]++;
      else if (l && COFF[l] > COFF[l+1] && COFF[l] > COFF[l-1] && d < MD)
        COFF[l]--;
    }
			
    for (int l = sl - 1; l >= 1; l--) COFF[l] -= COFF[l - 1];
			
    int lv = 1;
    int llv = 1;
			
    REP(l, sl) {
      data[p + l] = COFF[l] + MAX_DIFF;
      int t = l * 16 + lv * 4 + llv;
      COUNT[t][MAX_DIFF + COFF[l]]++;
      llv = lv;
      lv = max(0, min(3, 1 + COFF[l]));
    }
			
    p += sl;
  }
		
  REP(i, VF_TABLE + 1) re.encodeTable(i, COUNT[i]);
		
  p = 3;
  REP(x, sx * sy) {
    if (data[p] == MAX_VAL + 1) {
      p += sl;
      continue;
    }
			
    re.encode(VF_TABLE, AVG[x] / (SCALE * 4));
			
    int lv = 1;
    int llv = 1;
    REP(l, sl) {
      int v = data[p++];
      int t = l * 16 + lv * 4 + llv;
      re.encode(t, v);
      llv = lv;
      lv = max(0, min(3, 1 + v - MAX_DIFF));
    }
  }
		
  re.flush();
  REP(i, 256) re.encode<1>(0);
		
#ifdef REDUCE
  double r = (double)datavi.S / re.POS;
  while ((double)datavi.S / re.POS > r - REDUCE / 4.0) re.encode<1>(0), re.flush();
#endif
		
  rv.resize(re.POS / 4 + 1);
		
  return rv;
}
	
VI PsyhoDatCompression::decompress(const VI &data) {
  RangeDecoder rd((PUINT8)&data[0]);
		
  const int sx = rd.decode<10>();
  const int sy = rd.decode<10>();
  const int sl = rd.decode<10>();
		
  VI rvvi;
  rvvi.reserve((sx * sy * sl + 4) / 2 + 100);
  ((int**)&rvvi)[1] = ((int**)&rvvi)[0] + rvvi.capacity();
  PUINT16 rv = (PUINT16)&rvvi[0];
		
  rv[0] = sx;
  rv[1] = sy;
  rv[2] = sl;
		
  VI err;
  VI errt;
  while (true) {
    int type = rd.decode<1>();
    int hi = rd.decode<10>();
    int lo = rd.decode<10>();
    if (hi == sx && lo == sy) break;
    err.PB(hi * sy + lo);
    errt.PB(type);
  }
  err.PB(sx * sy);
  int errp = 0;
		
  const int SCALE = rd.decode<8>();
		
  VVVI sum(sx / SUM_STEP + 1, VVI(sy / SUM_STEP + 1, VI(sl, 0)));
  REP(x, sum.S) REP(y, sum[x].S) {
    sum[x][y][0] = 0;
    FOR(l, 1, sl) {
      int type = rd.decode<1>();
      int d;
      if (type == 0) {
        d = rd.decode<6>() - 32;
      } else {
        d = rd.decode<15>() - (1<<14);
      }
      sum[x][y][l] = sum[x][y][l-1] + d;
    }
  }
		
  //INT16 TC[MAX_SL][SHIFT_SIZE];    
  //REP(i, MAX_SL) REP(j, SHIFT_SIZE) TC[i][j] = rd.decode<8>();
		
  REP(l, MAX_TABLE) rd.decodeTable(l);
		
  int p = 3;
  REP(x, sx) REP(y, sy) { 
    int t = x * sy + y;
    if (t == err[errp]) {
      int type = errt[errp] ? MAX_VAL : 0;
      REP(i, sl) rv[p++] = type;
      errp++;
      continue;
    }
			
    int avg = rd.decode(VF_TABLE) * (SCALE * 4);
    VI &vs = sum[x / SUM_STEP][y / SUM_STEP];
			
    int off = 0;
    int lv = 1;
    int llv = 1;
    REP(l, sl) {
      int v = rd.decode(l * 16 + lv * 4 + llv);
      off += v - MAX_DIFF;
      rv[p++] = max(0, min(MAX_VAL, avg + vs[l] - off * SCALE));
      llv = lv;
      lv = max(0, min(3, 1 + v - MAX_DIFF));
    }
  }
		
  rvvi.resize((sx * sy * sl + 4) / 2);
		
  return rvvi;
		
}
	
