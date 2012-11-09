/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <sys/time.h>
#include <cstring>
#include <xmmintrin.h>
#include <cassert>
#include <malloc.h>
#include <stdint.h> 
#include "DelicatoDatCompression.h"
//#include "TraceChunk.h"
using namespace std;
#define NDEBUG

static const int MAX_L = 60;
static const int MAX_N = 910 * 910;

static const int NRANK_GOOD = 8;
static const int NRANK_BAD = 44;

static const int MIN_NBAD = 400;
static const int MAX_NBAD = 120000;
static const double MID_LIM = 40.0;
static const double HIGH_LIM = 100.0;
static const double MAX_AVG_ERR = 33.0;        

static const int NSAMPLING_OUTER = 6000; 
static const int NSAMPLING_INNER = 4; 

static const double LAGRANGE_STEP_SIZE = 3.0;

static const double AIM1    = 36.8;
static const double ACCEPT1 = 35.9;

static const double AIM2_HIGH = 35.8;
static const double AIM2_LOW  = 35.7;

static const double ACCEPT2 = 35.5;
static const double AIM3    = 35.5;

static const double AIM_BAIL = 35.3;

#define MEMSETZERO(x) memset(x, 0, sizeof(x))
#define TIME(str) for (bool __once = true; __once;) for (Timer __timer(str); __once; __once = false) 
#define rep(i, n) for (int i = 0; i < int(n); ++i)
#define rrep(i, n) for (int i = int(n) - 1; i >= 0; --i)
#define repeat(n) rep(__tmp__, n)
#define tr(it, c) for (typeof((c).begin()) it = (c).begin(); it != (c).end(); ++it )
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
#define all(c) (c).begin(), (c).end()

struct Timer {
    Timer(const string &str): str(str) { gettimeofday(&start, 0); }
    string str;
    timeval start;
	
    ~Timer() {
        timeval tmp;
        gettimeofday(&tmp, 0);
        const double sec = ((tmp.tv_sec - start.tv_sec) * 1000000 + tmp.tv_usec - start.tv_usec) / 1000000.0;
        cout << sec << " seconds; " << str << endl;
    }
};

template<int S>
struct FixedVector {
    double V[S];
    FixedVector() {}
    FixedVector(double t) { rep(i, S) V[i] = t; }
    double &operator[](int i) { return V[i]; }
    const double &cAt(int i) const { return V[i]; }
    
    double* begin() { return V; }
    double* end() { return V + S; }
};

typedef FixedVector<NRANK_GOOD + NRANK_BAD> FVD;

typedef pair<unsigned int, unsigned int> puu;
typedef vector<int> vi;
typedef vector<double> vd;
typedef vector<float> vf;
typedef pair<double, int> pdi;

typedef float v4sf __attribute__((vector_size(16), aligned(4*4)));
typedef int v4si __attribute__((vector_size(16), aligned(4*4)));
typedef uint32_t v4ui __attribute__((vector_size(16), aligned(4*4)));
typedef int32_t v2si __attribute__((vector_size(8), aligned(4*2)));
typedef long long v2di __attribute__((vector_size(16), aligned(4*4)));
typedef short v8hi __attribute__ ((vector_size (16)));

template<typename F, int S>
static double fold(F f, double t, const FixedVector<S> &A) {
    double res = t;
    rep(i, S) res = f(res, A.cAt(i));
    return res;
}

template<typename F, int S>
static FixedVector<S> apply(F f, const FixedVector<S> &A) {
    FixedVector<S> res;
    rep(i, S) res[i] = f( A.cAt(i));
    return res;
}

template<typename F, int S>
static FixedVector<S> apply(F f, const FixedVector<S> &A, const FixedVector<S> &B) {
    FixedVector<S> res;
    rep(i, S) res[i] = f(A.cAt(i), B.cAt(i));
    return res;
}

template<int S> static FixedVector<S> operator *(const FixedVector<S> &A, const FixedVector<S> &B)
{ return apply( multiplies<double>(), A, B ); }

template<int S> static FixedVector<S> operator /(const FixedVector<S> &A, const FixedVector<S> &B)
{ return apply( divides<double>(), A, B ); }

template<int S> static FixedVector<S> operator +(const FixedVector<S> &A, const FixedVector<S> &B)
{ return apply( plus<double>(), A, B ); }    

template<int S> static FixedVector<S> operator -(const FixedVector<S> &A, const FixedVector<S> &B)
{ return apply( minus<double>(), A, B ); }

template<int S> static FixedVector<S> operator *(const FixedVector<S> &A, double t)
{ return apply( bind1st(multiplies<double>(), t), A); }

template<int S> static FixedVector<S> operator +(const FixedVector<S> &A, double t)
{ return apply( bind1st(plus<double>(), t), A); }

template<int S> static FixedVector<S> operator /(const FixedVector<S> &A, double t)
{ return apply( bind2nd(divides<double>(), t), A); }

template<int S> static double sum(const FixedVector<S> &A) 
{ return fold( plus<double>(), double(0), A ); }

template<int S> static double scalarProduct(const FixedVector<S> &A, const FixedVector<S> &B) 
{ return sum(A * B); }

static float __inline__ round(float x) 
{ return x + (x >= 0.0f ? 0.5f: -0.5f); }

static int __inline__ iround(float x) 
{ return (int)round(x); }

uint64_t u64Round(double x) 
{ return (uint64_t)(0.5 + x); }

template<int MAX_NROWS>
struct LongVector {
    int64_t sum;
    int m;
    
    static const int MAX_NROWS4 = (MAX_NROWS+3) / 4;
    union {
        v4si V4[ MAX_NROWS4 ]; 
        int32_t i4[]; 
    };
    
    // make all values positive so we can use unsigned multiplication
    void calculateMinAndSum(int N) {
        const int E = (N + 3) / 4;
        for (int k = N; k < E * 4; ++k) i4[k] = 0;
        
        const int16_t INT16MIN = numeric_limits<int16_t>::max(); 
        union {
            v8hi V;
            int16_t i16[8];
        } minV = { {INT16MIN, 0, INT16MIN, 0, INT16MIN, 0, INT16MIN, 0} };
        
        sum = 0;
        for (int k = 0; k < E; ) {
            union {
                v4si V;
                int32_t i32[4];
            } sumV = {{0, 0, 0, 0}};
            const int EE = min(1<<14, E - k);    // we need to offload before it overflows
            for (int ii = 0; ii < EE; ++ii, ++k) {
                const v4si VK = V4[k];
                sumV.V += VK;
                minV.V = __builtin_ia32_pminsw128( minV.V, (v8hi)VK );
            }
            sum += (int64_t)sumV.i32[0] + (int64_t)sumV.i32[1] + (int64_t)sumV.i32[2] + (int64_t)sumV.i32[3];
        }
        m = min( min(minV.i16[0], minV.i16[2]), min(minV.i16[4], minV.i16[6]) );
        const v4si mV = {m, m, m, m};
        rep(k, MAX_NROWS4) V4[k] -= mV;
        for (int k = N; k < E * 4; ++k) i4[k] = 0;
    }
} __attribute__((aligned(4*4))) ;

union i4vector {
    i4vector() { const v4ui tmp = {0, 0, 0, 0}; v4 = tmp; } 
    v4ui v4;
    uint32_t i4[4];
    int64_t sum() { return int64_t( i4[0] ) + i4[1] + i4[2] + i4[3]; }
} ;

// helper function for scalarProd
void static inline __update(i4vector &lowSum, i4vector &highSum, const v8hi &X, const v8hi &Y) {
    const v8hi LOW_PART  = __builtin_ia32_pmullw128( X,Y );
    const v8hi HIGH_PART = __builtin_ia32_pmulhuw128(X, Y );
	
    const v8hi VZERO = {0,0,0,0,0,0,0,0};
    lowSum.v4 += (v4si) __builtin_ia32_punpcklwd128( LOW_PART, VZERO )
	+ (v4si) __builtin_ia32_punpckhwd128( LOW_PART, VZERO );
	
    highSum.v4 += (v4si) __builtin_ia32_punpcklwd128( HIGH_PART, VZERO )
	+ (v4si) __builtin_ia32_punpckhwd128( HIGH_PART, VZERO );    
}

// computes (A*C), (A*D), (B*C), (B*D)
//typedef pair<int64_t, int64_t> pi64i64;
template<int MAX_NROWS>
static FixedVector<4> scalarProd(int N, const LongVector<MAX_NROWS> &A, const LongVector<MAX_NROWS> &B, const LongVector<MAX_NROWS> &C, const LongVector<MAX_NROWS> &D) {
    const int N4 = (N+3)/4;
    int64_t AC = 0, AD = 0, BC = 0, BD = 0;
    
    for (int i = 0; i < N4; ) {
        i4vector lowPartAC, lowPartAD, lowPartBC, lowPartBD;
        i4vector highPartAC, highPartAD, highPartBC, highPartBD;
		
        const int E = min( N4 - i, 32768 );
        rep(ii, E) {    // we need to offload before it overflows
            const v8hi AI = (v8hi)A.V4[i];
            const v8hi BI = (v8hi)B.V4[i];
            const v8hi CI = (v8hi)C.V4[i];
            const v8hi DI = (v8hi)D.V4[i];
            
            __update(lowPartAC, highPartAC, AI, CI);
            __update(lowPartAD, highPartAD, AI, DI);
            __update(lowPartBC, highPartBC, BI, CI);
            __update(lowPartBD, highPartBD, BI, DI);
			
            ++i;
        }
        
        AC += (highPartAC.sum() << 16) + lowPartAC.sum();   
        AD += (highPartAD.sum() << 16) + lowPartAD.sum();   
        BC += (highPartBC.sum() << 16) + lowPartBC.sum();   
        BD += (highPartBD.sum() << 16) + lowPartBD.sum();   
    }
    
    FixedVector<4> FVres;
    FVres[0] = double( AC + A.m * C.sum + C.m * A.sum - int64_t(N) * int64_t(A.m) * int64_t(C.m) );
    FVres[1] = double( AD + A.m * D.sum + D.m * A.sum - int64_t(N) * int64_t(A.m) * int64_t(D.m) );
    FVres[2] = double( BC + B.m * C.sum + C.m * B.sum - int64_t(N) * int64_t(B.m) * int64_t(C.m) );
    FVres[3] = double( BD + B.m * D.sum + D.m * B.sum - int64_t(N) * int64_t(B.m) * int64_t(D.m) );
    return FVres;
}  

class BitOutputStream {
public:
    BitOutputStream(): idx(0), V(3000000, 0) {}
    
    void alignIdx() {
        idx += (32 - idx % 32) % 32;
    }
    
    template<typename T>
    void put(const T &t, int n = sizeof(T) * 8) {
        union {
            T t;
            uint64_t u;
        } tmp = { t };    
		
        rrep(i, n) {
            int b = (tmp.u >> i) & 1;
            pushBit( b  );
        }
    }
	
    void put32(uint32_t u) {
        V[idx / 32] = u;
        idx += 32;
    }
	
    int getPosition() const { 
        return idx;
    }
	
    template<typename T>
    void inject(int pos, const T &t, int n = sizeof(T) * 8) {
        int tmp = idx;
        idx = pos;
        put<T>(t, n);
        idx = tmp;
    }
	
    vector<int> &data() {
        V.resize( (idx + 32) / 32 );
        return V;
    }

    void pushBit(uint32_t b) {     
        int bitIdx = 31 - (idx & 31);        
        V[idx / 32] |= b << bitIdx;
        ++idx;
    }        
private:
    unsigned int idx;
    vector<int> V;    
    

} ;

class BitInputStream {
public:
    BitInputStream(const vector<int> &underlying_vector): V(underlying_vector), idx(0) {}
	
    void alignIdx() {
        idx += (32 - idx % 32) % 32;
    }
    
    void setPosition(int i) {
        idx = i;
    }
    
    template<typename T>
    T get() {
        union {
            uint64_t u;        
            T t;
        } tmp = { getBits( sizeof(T) * 8 ) };
		
        return tmp.t;
    }
	
    uint32_t get32() {
        uint32_t res = V[idx / 32];
        idx += 32;
        return res;
    }
    
    uint64_t getBits(int n) {
        uint64_t v = 0;
        rep(i, n) {
            v = (v << 1) | popBit();
        } 
        return v;
    }
	
private:    
    const vector<int> &V;
    unsigned int idx;    
    
    int popBit() {       
        const int bitIdx = 31 - (idx & 31);
        const int b = (V[idx / 32] >> bitIdx) & 1;    
        ++idx;
        return b;
    }    
};

const int LookUpSize = 1024;
class Alphabet {
public:
    Alphabet(const vi &frequenceTable): n(frequenceTable.size()), intervalLookup(n) {
        int freqSum = 0;
        tr(it, frequenceTable) freqSum += *it;
        const double FACTOR = pow(2.0, 32);
        
        uint64_t cum = 0;
        unsigned int u = 0;
        rep(i, n) {
            if ( frequenceTable[i] > 0 ) {
                cum += frequenceTable[i];
                uint64_t v = u64Round( (cum * FACTOR) / freqSum );
                intervalLookup[i] = puu(u+1, v-u-2);  // 0, 1
                u = v;
                symbolLookup.push_back( pair<double, int>(v / FACTOR, i) );            
            }
        }
        
        int j = 0;
        rep(i, LookUpSize) {
            double fraction = double(i) / LookUpSize; // [0, 1[
            for (; ; ++j) {
                if ( symbolLookup[j].first > fraction ) {
                    fastLookUp[i] = j;
                    break;
                }
            }            
        }
    }
	
    puu __inline__ getInterval(int symbol) const {
        return intervalLookup[symbol];
    }
	
    // finds the interval covering val and returns the corresponding symbol
    int __inline__ getSymbol(double fraction) const {
        for (int i = fastLookUp[ int(fraction * LookUpSize) ]; ; ++i) {
            if ( symbolLookup[i].first > fraction ) return symbolLookup[i].second;
        }
    }
	
private:
    int n;    
    vector< puu > intervalLookup;
    vector< pdi > symbolLookup;
    uint16_t fastLookUp[LookUpSize];
};

union U64 {
    uint64_t u64;
    struct {
        uint32_t low, high;
    };    
};

static void encode(Alphabet &A, int n, const int str[], BitOutputStream &bout) {   
    int position = bout.getPosition();    
    bout.put<int>(0); // dummy, will be overwritten later
    
    bout.alignIdx();
	
    U64 L, R;
    L.u64 = 0;
    R.u64 = ~(0ULL); 
	
    uint64_t bits = 0;
    int nBits = 0;
	
    rep(i, n) {
        const puu interval = A.getInterval( str[i] );
        const uint64_t fi = interval.first;
        const uint64_t se = interval.second;
		
        const U64 LB = { R.low * fi };
        const U64 RB = { R.low * se };
        
        L.u64 += R.high * fi + LB.high;
        R.u64 = R.high * se + RB.high;    
		
        const U64 XOR64 = { L.u64 ^ (L.u64 + R.u64) };
        
        if ( unlikely(XOR64.high == 0) ) {        
            // -------- first 32 bits ----------
            bits = (bits << 32) | L.high;
            bout.put32( bits >> nBits );    
            // ---------------------------------
            
            const int cnt = __builtin_clz( XOR64.low ); // todo: anta att detta funkar?        
            if ( cnt > 0 ) {            
                bits = (bits << cnt) | (L.low >> (32 - cnt));
                nBits += cnt;    
				
                L.u64 = ((L.u64 << 32) | LB.low) << cnt; // padda med ettor istället?
                R.u64 = ((R.u64 << 32) | RB.low) << cnt;    
            }
        }
        else {
            const int cnt = __builtin_clz( XOR64.high );
            if ( likely(cnt > 0) ) {
                bits = (bits << cnt) | ( L.u64 >> (64 - cnt) );
                nBits += cnt;
                L.u64 = (L.u64 << cnt) | (LB.low >> (32 - cnt));
                R.u64 = (R.u64 << cnt) | (RB.low >> (32 - cnt));                
            }
        }
		
        if ( unlikely(nBits >= 32) ) {
            bout.put32( bits >> (nBits - 32) );
            nBits -= 32;
        }
    }
	
    if ( nBits > 0 ) {
        bout.put<uint64_t>( bits, nBits );
    }
    bout.put<uint64_t>( 1, 1 );
    bout.put<uint64_t>( 0, 31 );
    bout.put<uint64_t>( 0, 64 );
	
    bout.inject<int>(position, bout.getPosition() ); // <-- overwrite the dummy value here
}

static void decode(const Alphabet &A, int n, BitInputStream &bin, int out[]) {
    const int nextBitIdx = bin.get<int>();
    bin.alignIdx();
	
    U64 L, R;
    L.u64 = 0;
    R.u64 = ~(0ULL); 
	
    int nBits = 0;
    
    uint64_t u = 0, u2 = 0;
    repeat(4) {
        u = (u << 32) | (u2 >> 32);
        u2 = (u2 << 32) | bin.get32();    
    }
    
    rep(i, n) {
        const uint64_t uTmp = (nBits == 0 ? u: (u << nBits) | (u2 >> (64 - nBits)));
        const int symbol = A.getSymbol( double(uTmp - L.u64) / R.u64 );
        out[i] = symbol;
        const puu interval = A.getInterval(symbol);    
        const uint64_t fi = interval.first;
        const uint64_t se = interval.second;
		
        const U64 LB = { R.low * fi };
        const U64 RB = { R.low * se };
        
        L.u64 += R.high * fi + LB.high;
        R.u64 = R.high * se + RB.high;    
		
        const U64 XOR64 = { L.u64 ^ (L.u64 + R.u64) };
        
        if ( unlikely(XOR64.high == 0) ) {
            // -------- first 32 bits ----------
            u = (u << 32) | (u2 >> 32);
            u2 = (u2 << 32) | bin.get32();            
            // ---------------------------------
            
            const int cnt = __builtin_clz( XOR64.low ); // todo: anta att detta funkar?
            if ( cnt > 0 ) {
                nBits += cnt;    
                
                L.u64 = ((L.u64 << 32) | LB.low) << cnt; // padda med ettor istället?
                R.u64 = ((R.u64 << 32) | RB.low) << cnt;    
            }
        }        
        else {
            const int cnt = __builtin_clz( XOR64.high );
            if ( likely(cnt > 0) ) {
                nBits += cnt;
                L.u64 = (L.u64 << cnt) | (LB.low >> (32 - cnt));
                R.u64 = (R.u64 << cnt) | (RB.low >> (32 - cnt));                
            }
        }
        
        if ( unlikely(nBits >= 32) ) {
            u = (u << 32) | (u2 >> 32);
            u2 = (u2 << 32) | bin.get32();        
            nBits -= 32;
        }
    }
	
    bin.setPosition( nextBitIdx );
}

// OBS, changes data!
static void storeInts(int n, int data[], BitOutputStream &bout) { 
    int m = data[0];
    int M = data[0];
    rep(i, n) {
        const int tmp = data[i];
        m = min(m, tmp);
        M = max(M, tmp);
    }
	
    vi freq(1 + M-m, 0);
    rep(i, n) {
        const int tmp = data[i];
        ++freq[ tmp - m ];
        data[i] = tmp - m;
    }
    int maxFreq = *max_element( all(freq) );
    int k = 1;
    while (maxFreq >= (1 << k)) ++k;
    bout.put<int>( m );
    bout.put<int>( M );    
    bout.put<int>( k );
    
    tr(it, freq) bout.put<int>( *it, k );
    Alphabet A(freq);
    encode(A, n, data, bout);
    // rep(i, n) data[i] += m;   // shave off a few ms by not restoring
}

static void loadInts(int N, BitInputStream &bin, int data[]) {
    int m = bin.get<int>();
    int M = bin.get<int>();
    int k = bin.get<int>();
    
    vi freq(1 + M-m, 0);
    
    tr(it, freq) *it = (int)bin.getBits(k);
	
    Alphabet A(freq);
    decode(A, N, bin, data);
    
    rep(i, N) data[i] += m;
}

template<int RANK, int MAX_NROWS>
class RowExtractor {
public:
    void init(int L_, int rank_) {
        idx = 0;
        L = L_;
        rank = rank_;
        MEMSETZERO(base);
    }
    
    void load(BitInputStream &bin, int nScore, int L, int nRows) {
        rep(i, 15) {
            rep(j, nScore) {
                rep(k, 4) {
                    const int x = i * 4 + k;
                    if ( x >= L ) continue;
                    base[i][j][k] = bin.get<float>();                
                }
            }
        }
        rep(i, nScore) {
            loadInts( nRows, bin, score[i] );
        }        
    }
	
    void extractNextRow(int nScore, int L, void *destination) {
        const v4sf VZERO = {0.0f, 0.0f, 0.0f, 0.0f};
		
        v4sf scoreV4[RANK];
        rep(j, nScore) {
            const float s = score[j][idx];
            const v4sf S = {s, s, s, s};
            scoreV4[j] = S;
        }
		
        v8hi rowDataBuffer[8];
        rep(x, 8) {
            v4sf sum1 = VZERO; 
            v4sf sum2 = VZERO; 
			
            rep(j, nScore) {
                sum1 += scoreV4[j] * baseV4[2*x+0][j];
                sum2 += scoreV4[j] * baseV4[2*x+1][j];
            }
			
            const v4sf max4f = {16383.0f, 16383.0f, 16383.0f, 16383.0f};                
            sum1 = __builtin_ia32_maxps(VZERO, __builtin_ia32_minps(max4f, sum1) );
            sum2 = __builtin_ia32_maxps(VZERO, __builtin_ia32_minps(max4f, sum2) );
			
            rowDataBuffer[x] = __builtin_ia32_packssdw128(
														  __builtin_ia32_cvtps2dq(sum1),
														  __builtin_ia32_cvtps2dq(sum2));
        }
        
        memcpy ( destination, rowDataBuffer, 2 * L );
        ++idx;
    }
    
private:
    int idx, L, rank;
	
    union {
        v4sf baseV4[16][ RANK ];
        float base[16][  RANK ][4];        
    };    
    
    int32_t score[RANK][MAX_NROWS];    
};

static vector<vd> gaussReduce(vector<vd> M, vector<vd> R) {   
    const int n = M.size();
    assert(n == (int)R.size());
    const int m = R[0].size();
	
    rep(i, n) {
        int pivot = i;
        
        for (int row = i; row < n; ++row) {
            if ( abs( M[pivot][i] ) < abs( M[row][i] ) ) pivot = row;
        }
        if ( M[pivot][i] == 0.0 ) continue;
        rep(j, n) swap( M[pivot][j], M[i][j] );
        rep(j, m) swap( R[pivot][j], R[i][j] );
		
        double d = M[i][i];
        for (int x = i; x < n; ++x) {
            M[i][x] /= d;
        }
        rep(x, m) R[i][x] /= d;
		
        for (int row = i + 1; row < n; ++row) {
            double d = M[row][i];
            for (int x = i; x < n; ++x) {
                M[row][x] -= d * M[i][x];
            }
            rep(x, m) {
                R[row][x] -= d * R[i][x];
            }
        }
    }
	
    rrep(i, n) {
        rep(row, i) {
            rep(x, m) {
                R[row][x] -= R[i][x] * M[row][i];
            }
            M[row][i] = 0.0;
        }
    }
    
    return R;
}

class TransposeSum {
public:   
    TransposeSum(int L_) {
        L  = L_;
        cnt = 0;
        MEMSETZERO(T4);
        MEMSETZERO(T6);
        MEMSETZERO(offload);
    }
    
    void __inline__ pushRow(const uint16_t *const ptr) {
        if ( unlikely( (++cnt & 32767) == 0) ) offloadToBuffer();
        const v4si VZERO =  {0,0,0,0};
        
        union {
            v8hi rowV8[8];
            uint16_t row16[64];
        } __attribute__ ((aligned(4*4)));
        memcpy( rowV8, ptr, L * 2 );
		
        int idx = 0;
        rep(i, L) {
            const uint16_t rowI = row16[i];
            const v8hi rowI8 = {rowI, rowI, rowI, rowI, rowI, rowI, rowI, rowI};
            
            const int E = i/8 + 1;
            rep(j, E) {
                const v8hi rowJ8 = rowV8[j];
                const v8hi LOW_PART = __builtin_ia32_pmullw128( rowI8, rowJ8 );
                const v8hi HIGH_PART = __builtin_ia32_pmulhuw128( rowI8, rowJ8 );
                
                T5[idx+0] += (v4si) __builtin_ia32_punpcklwd128( LOW_PART, (v8hi)VZERO );
                T5[idx+1] += (v4si) __builtin_ia32_punpckhwd128( LOW_PART, (v8hi)VZERO );
				
                T5[idx+2] += (v4si) __builtin_ia32_punpcklwd128( HIGH_PART, (v8hi)VZERO );            
                T5[idx+3] += (v4si) __builtin_ia32_punpckhwd128( HIGH_PART, (v8hi)VZERO );
                
                idx += 4;
            }
        }
    }
	
    vector< vf >  calculateEigenVectors() {
        double A[60][60];    
        offloadToBuffer();
		
        double Q[60][60];
        rep(i, L) rep(j, L) Q[i][j] = (i == j ? 1.0: 0.0);
        
        double buff1[60];
        double buff2[60];
        double buff3[60];
        double buff4[60];    
        
        int idx = 0;
        rep(i, L) {
            const int E = i/8 + 1;
            rep(j, E) {
                rep(k, 8) {
                    int J = j*8+k;
                    if ( J >= L ) continue;            
                    A[J][i] = A[i][J] = double( offload[idx + k] );
                }
                idx += 8;
            }
        }        
        
        TIME("Jacobi eigenvalues") rep( iOuter, 500 ) {
            rep(i, L) {
                double maxVal = -1.0;
                int j = 0;
                rep(x, L) if ( x != i ) {
                    double testVal = abs( A[i][x] );
                    if ( testVal > maxVal ) {
                        maxVal = testVal;
                        j = x;
                    }
                }
                if ( maxVal < 0.001 ) continue;
				
                const double a = A[i][i];
                const double b = A[i][j];
                const double d = A[j][j];
                
                const double k = (a - d) / (2.0 * b);
                const double t = k - sqrt(1.0 + k*k);
                
                const double c = 1.0 / sqrt(1.0 + t*t);
                const double s = c * t;
				
                rep(ii, L) buff1[ii] = A[ii][i];
                rep(ii, L) buff2[ii] = A[ii][j];
                
                rep(ii, L) buff3[ii] = buff1[ii] * c - s * buff2[ii];
                rep(ii, L) buff4[ii] = buff1[ii] * s + c * buff2[ii];
                
                const double h1 = c * buff3[i] - s * buff3[j];
                const double h4 = s * buff4[i] + c * buff4[j];
                
                buff3[i] = h1;
                buff3[j] = 0.0;
                buff4[i] = 0.0;
                buff4[j] = h4;
                
                rep(ii, L) A[ii][i] = buff3[ii];
                rep(ii, L) A[ii][j] = buff4[ii];
                
                rep(ii, L) A[i][ii] = buff3[ii];
                rep(ii, L) A[j][ii] = buff4[ii];
                
				
                rep(ii, L) buff1[ii] = Q[ii][i];
                rep(ii, L) buff2[ii] = Q[ii][j];
                
                rep(ii, L) Q[ii][i] = buff1[ii] * c - s * buff2[ii];
                rep(ii, L) Q[ii][j] = buff1[ii] * s + c * buff2[ii];  
            }
        }
		
        vector< pdi > V(L);
        rep(i, L) V[i] = pdi( abs(A[i][i]), i);    // eigenvalues on diagonal
        sort( all(V), greater< pdi >() );   // sort on eigenvalues
		
        vector< vf > res(L, vf(L));
        rep(i, L) {
            rep(j, L) {
                res[i][j] = Q[i][ V[j].second  ];
            }
        }
        
        return res;        
    }
    
private:
    int L;
    int cnt;
    
    union {
        v2di T3[1024];
        int64_t T4[2048];
    };    
	
    union {
        v4si T5[60*128/4];    // 64 * 2 tal
        uint32_t T6[60*128];
    };
    
    uint64_t offload[60 * 64];
	
    void offloadToBuffer() {
        rep(i, 450) {
            rep(k, 8) {
                offload[i*8+k] += uint64_t(T6[i * 16 + k%8 + 0]) + ( uint64_t( T6[i * 16 + k%8 + 8] ) << 16);    
            }
        }
		
        MEMSETZERO(T6);
    }    
};

template<int RANK, int MAX_NROWS>
struct ScoreCalc {
    union {
        float base[16][  RANK ][4];
        v4sf baseV4[16][ RANK ];
    };
	
    uint64_t length[MAX_NROWS];
	
    union {
        float minScore[RANK];
        v4sf minScoreV4[RANK/4];
    };
	
    union {
        float maxScore[RANK];
        v4sf maxScoreV4[RANK/4];
    };
	
    union {
        float score[MAX_NROWS][RANK];
        v4sf scoreV4[MAX_NROWS][RANK/4];
    };
	
    int N, L;
    const uint16_t *ptr;
    
    void init(const vector< vector<float> > &base_, const uint16_t *const ptr_) {
        N = 0;
        L = base_.size();
        ptr = ptr_;
        MEMSETZERO(base);
        rep(i, 15) {
            rep(j, RANK) {
                rep(k, 4) {
                    const int y = i * 4 + k;
                    base[i][j][k] = (y >= L || j >= L ? 0.0f: base_[y][j]);
                }
            }
        }        
    }
	
    void __inline__ pushRow(const uint16_t *const ptr) {    
        const v4si VZERO = {0,0,0,0};
        
        v4si rowNorm2 = VZERO;
        v4sf tmpScore[RANK];
        rep(i, RANK) tmpScore[i] = (v4sf)VZERO;
		
        const int E = (L - 1) / 8;
        rep(i, E) {
            v8hi rowI = (v8hi) _mm_loadu_si128( (__m128i*)(ptr + i*8) );
            
            const v4si rowIA = (v4si) __builtin_ia32_punpcklwd128( rowI, (v8hi)VZERO );
            rowNorm2 += rowIA * rowIA;
            const v4sf rowIAF = __builtin_ia32_cvtdq2ps( rowIA );
            
            const v4sf *baseA = baseV4[i*2+0];
            rep(ii, RANK) tmpScore[ii] += rowIAF * baseA[ii];        
            
            const v4si rowIB = (v4si) __builtin_ia32_punpckhwd128( rowI, (v8hi)VZERO );
            rowNorm2 += rowIB * rowIB;
            const v4sf rowIBF = __builtin_ia32_cvtdq2ps( rowIB );
            
            const v4sf *baseB = baseV4[i*2+1];
            rep(ii, RANK) tmpScore[ii] += rowIBF * baseB[ii];                
        }
        union rowIUnion_t { v8hi rowI; uint16_t rowIu16[8]; } rowIUnion;
        int k, ii = E*8;
        for (k = 0; ii < L; ++k, ++ii) rowIUnion.rowIu16[k] = ptr[ii];
        for (; k < 8; ++k) rowIUnion.rowIu16[k] = 0;            
        
        const v4si rowIA = (v4si) __builtin_ia32_punpcklwd128( rowIUnion.rowI, (v8hi)VZERO );
        rowNorm2 += rowIA * rowIA;
        const v4sf rowIAF = __builtin_ia32_cvtdq2ps( rowIA );
		
        const v4sf *baseA = baseV4[E*2+0];
        rep(ii, RANK) tmpScore[ii] += rowIAF * baseA[ii];        
		
        const v4si rowIB = (v4si) __builtin_ia32_punpckhwd128( rowIUnion.rowI, (v8hi)VZERO );
        rowNorm2 += rowIB * rowIB;
        const v4sf rowIBF = __builtin_ia32_cvtdq2ps( rowIB );
		
        const v4sf *baseB = baseV4[E*2+1];
        rep(ii, RANK) tmpScore[ii] += rowIBF * baseB[ii];          
		
        union {
            v2di d;
            uint64_t u[2];
        } tmp = {  __builtin_ia32_paddq128( 
										   (v2di)__builtin_ia32_punpckldq128( (v4si)rowNorm2, VZERO),
										   (v2di)__builtin_ia32_punpckhdq128( (v4si)rowNorm2, VZERO) ) };
        
        length[N] = tmp.u[0] + tmp.u[1];
        
        rep(i, RANK/4) {
            scoreV4[N][i] = __builtin_ia32_haddps( 
                                                  __builtin_ia32_haddps( tmpScore[i*4+0], tmpScore[i*4+1] ),
                                                  __builtin_ia32_haddps( tmpScore[i*4+2], tmpScore[i*4+3] ) );        
        }
		
        ++N;
    }
	
    void calculateMinAndMax() {
        rep(j, RANK) minScore[j] = numeric_limits<float>::max();
        rep(j, RANK) maxScore[j] = numeric_limits<float>::min();
        
        rep(i, N) {
            rep(j, RANK/4) {
                const v4sf s = scoreV4[i][j];
                minScoreV4[j] = __builtin_ia32_minps( minScoreV4[j], s );
                maxScoreV4[j] = __builtin_ia32_maxps( maxScoreV4[j], s );    
            }
        }
    }
    
    vi getBadIdx() {    
        Timer timer("calc nBad");
        double sumGood = 0.0;        
        vd V, W(N);
        V.reserve(910*910);        
        const double rL = 1.0 / L;
        const double SUM_LIM = MAX_AVG_ERR * N;
        
        int highCnt = 0;
        rep(i, N) {
            double diff = length[i];
            rep(j, RANK) {
                const double tmp = score[i][j];
                diff -= tmp * tmp;
            }
            diff *= rL;
            W[i] = diff;
            
            if ( likely(diff < MID_LIM) ) {
                sumGood += diff;
            }
            else if ( likely(diff < HIGH_LIM ) ) {
                V.push_back( diff );
            }
            else {
                ++highCnt;
            }
        }
        if ( highCnt == 0 && V.size() == 0 ) return vi();
        
        cout << "highCnt: " << highCnt << ", V.size() : " << V.size() << endl;
        double pivot;
        if ( highCnt >= MAX_NBAD ) {
            pivot = HIGH_LIM;
        }
        else {            
            sort( all(V) );
            const int E = V.size() + min(highCnt - MIN_NBAD, 0);
			
            int i = 0;
            for (; i < E && sumGood < SUM_LIM; ++i) {
                sumGood += V[i];
            }
            i = max(i, int(highCnt + V.size() - MAX_NBAD) );
            pivot = i >= (int)V.size() ? HIGH_LIM: V[i];
            
            while (true) {
                ++i;
                if (i >= (int)V.size()) {
                    pivot = HIGH_LIM;
                    break;
                }
                else if ( pivot != V[i] ) {
                    pivot = V[i];
                    break;
                }
            }
        }
        cout << "pivot: " << pivot << endl;
        
        vi res;
        res.reserve(MAX_NBAD);
        rep(i, N) {
            if ( unlikely( W[i] >= pivot ) ) {
                res.push_back( i );
            }
        }
		
        return res;
    }
    
    double calcOrthogonalErrorSum() {
        double res = 0.0;
        
        rep(i, N) {
            double diff = length[i];
            rep(j, RANK) {
                const double tmp = score[i][j];
                diff -= tmp * tmp;
            }
            res += diff;
        }
        
        return res;
    }    
    
    FixedVector<RANK> calculateError( FixedVector<RANK> &factorV ) const {
        const v4sf VZERO = {0.0f, 0.0f, 0.0f, 0.0f};
        union errU_t {
            v4sf errSumV4[ RANK/4  ];
            float errSumf4[ RANK  ];        
        } errU;
        rep(i, RANK/4) errU.errSumV4[i] = VZERO;
		
        union {
            v4sf V4[RANK/4];
            float F4[RANK];
        } factor, rcFactor;
		
        rep(j, RANK) {
            factor.F4[j] = factorV[j];
            rcFactor.F4[j] = 1.0 / factorV[j];
        }
        
        rep(i, N) {
            rep(j, RANK/4) {
                const v4sf s = scoreV4[i][j+0];
                const v4si roundedToInt =  __builtin_ia32_cvtps2dq( s * rcFactor.V4[j] );        
                const v4sf roundedAsFloat = __builtin_ia32_cvtdq2ps( roundedToInt );
                const v4sf err = roundedAsFloat * factor.V4[j] - s;
                errU.errSumV4[j] += err * err;
            }
        }
        FixedVector<RANK> res;
        rep(i, RANK) res[i] = errU.errSumf4[i];
        return res;
    }    
    
    FixedVector<RANK> calculateSize( FixedVector<RANK> &factorV ) const {        
        if ( N == 0 ) return 0.0;
        
        union {
            v4sf V4[RANK/4];
            float F4[RANK];
        }  rcFactor;
        rep(j, RANK) {
            rcFactor.F4[j] = 1.0 / factorV[j];
        }
        
        int minVal[RANK];
        int maxVal[RANK];
        vi freq[RANK];
        
        rep(j, RANK) {        
            minVal[j] = -1 + iround(minScore[j] * rcFactor.F4[j] );
            maxVal[j] =  1 + iround(maxScore[j] * rcFactor.F4[j] );
            
            int sz = maxVal[j] - minVal[j] + 1;
            freq[j].resize( sz, 0 );        
        }
        
        rep(i, N) {
            rep(j, RANK/4) {
                const v4sf s = scoreV4[i][j];
				
                union {
                    v4si v4;
                    int i4[4];
                } roundedToInt = { __builtin_ia32_cvtps2dq( s * rcFactor.V4[j] ) };
				
                rep(k, 4) {
                    const int J = j*4 + k;
                    freq[ J ][ roundedToInt.i4[k] - minVal[J] ] += 1;
                }
            }
        }
		
        FixedVector<RANK> res;
        double rN = 1.0 / N;
        const double log2 = log(2.0);
        rep(j, RANK) {
            double sum = 128.0; // overhead //
			
            tr(it, freq[j]) {
                if ( unlikely( *it > 0 ) ) {
                    const double f = *it;
                    sum -= f * log(f * rN);
                }
            }
            
            res[j] = sum / log2;
        }
        
        
        return res;
    }
    
    void saveBasis(BitOutputStream &bout) const {
        rep(i, 15) {
            rep(j, RANK) {
                rep(k, 4) {
                    const int x = i * 4 + k;
                    if ( x >= L ) continue;
                    bout.put<float>( base[i][j][k]  );
                }
            }
        }    
    }
	
    void saveScores(BitOutputStream &bout, const vd &factor) const {
        vi data(N);
        
        rep(i, RANK) {
            const double rcFactor = 1.0 / factor[i];
            rep(j, N) {
                data[j] = iround( score[j][i] * rcFactor );    
            }
            storeInts(N, &data.front(), bout);
        }
    }    
	
    void updateBasis(const vd &factor) {
        assert( factor.size() == RANK );
        rep(i, 15) {
            rep(j, RANK) {
                rep(k, 4) {
                    const int x = i * 4 + k;
                    if ( x >= L ) continue;
                    base[i][j][k] *= factor[j];
                }
            }
        }    
    }
    
    uint64_t calculateExactError(const vd &factor, const vi &rowIdx) {
        Timer timer("Matrix multiply");
        vd rcFactor(RANK);
        rep(i, RANK) rcFactor[i] = 1.0 / factor[i];        
        uint64_t sum = 0;
        
        rep(i, N) {
            const v4sf VZERO = {0.0f, 0.0f, 0.0f, 0.0f};
            const v4sf max4f = {16383.0f, 16383.0f, 16383.0f, 16383.0f};
            v4sf scoreV4[RANK];
            rep(j, RANK) {
                const float s = round( score[i][j] * rcFactor[j] );
                const v4sf S = {s, s, s, s};
                scoreV4[j] = S;
            }        
            const uint16_t *rowPtr = ptr + 3 + L * rowIdx[i];
			
            union {
                v4si V;
                uint32_t U32[4];
            } diffSumRow = { (v4si)VZERO };
            
            const int E = (L - 1) / 8;
            rep(x, E) {
                v4sf sum1 = VZERO, sum2 = VZERO; 
				
                rep(j, RANK) {
                    sum1 += scoreV4[j] * baseV4[2*x+0][j];
                    sum2 += scoreV4[j] * baseV4[2*x+1][j];
                }
				
                sum1 = __builtin_ia32_maxps(VZERO, __builtin_ia32_minps(max4f, sum1) );
                sum2 = __builtin_ia32_maxps(VZERO, __builtin_ia32_minps(max4f, sum2) );
				
                const v8hi realRow = (v8hi) _mm_loadu_si128( (__m128i*)rowPtr );
                rowPtr += 8;
				
                const v4si diffSum1 = __builtin_ia32_cvtps2dq(sum1) - (v4si)__builtin_ia32_punpcklwd128( realRow, (v8hi)VZERO );
                const v4si diffSum2 = __builtin_ia32_cvtps2dq(sum2) - (v4si)__builtin_ia32_punpckhwd128( realRow, (v8hi)VZERO );
                diffSumRow.V += diffSum1 * diffSum1 + diffSum2 * diffSum2;
            }
            v4sf sum1 = VZERO, sum2 = VZERO; 
			
            rep(j, RANK) {
                sum1 += scoreV4[j] * baseV4[2*E+0][j];
                sum2 += scoreV4[j] * baseV4[2*E+1][j];
            }
			
            sum1 = __builtin_ia32_maxps(VZERO, __builtin_ia32_minps(max4f, sum1) );
            sum2 = __builtin_ia32_maxps(VZERO, __builtin_ia32_minps(max4f, sum2) );
			
            union { v8hi V; uint16_t U16[8]; } realRow;
            int k = 0;
            const int EE = L - E * 8;
            for (; k < EE; ++k) realRow.U16[k] = rowPtr[k];
            for (; k < 8; ++k) realRow.U16[k] = 0;
            
            const v4si diffSum1 = __builtin_ia32_cvtps2dq(sum1) - (v4si)__builtin_ia32_punpcklwd128( realRow.V, (v8hi)VZERO );
            const v4si diffSum2 = __builtin_ia32_cvtps2dq(sum2) - (v4si)__builtin_ia32_punpckhwd128( realRow.V, (v8hi)VZERO );
            diffSumRow.V += diffSum1 * diffSum1 + diffSum2 * diffSum2;          
			
            sum += (diffSumRow.U32[0] + diffSumRow.U32[1]) + (diffSumRow.U32[2] + diffSumRow.U32[3]);
        }
        
        return sum;    
    }
    
    void updateToOptimalBasis(const vd &factor, const vi &rowIdx) {
        if ( N < 300 ) return; // rank too low, not stable
        assert( factor.size() == RANK );
		
        LongVector<MAX_NROWS> *dataV = (LongVector<MAX_NROWS>*)memalign(16, sizeof(LongVector<MAX_NROWS>) * L); 
        LongVector<MAX_NROWS> *scoreV = (LongVector<MAX_NROWS>*)memalign(16, sizeof(LongVector<MAX_NROWS>) * RANK);
        
        TIME("Transposing data") {
            rep(i, N) {
                const uint16_t* rowPtr = ptr + 3 + L * rowIdx[i];
                rep(ii, L) {
                    dataV[ii].i4[i] = rowPtr[ii];
                }
            } 
        }
        TIME("min/sum") rep(ii, L) dataV[ii].calculateMinAndSum(N);
		
        updateBasis( factor );
        vd rcFactor( factor );
        tr(it, rcFactor) *it = 1.0 / *it;
        
        TIME("Transposing scores") {
			rep(i, N) {
				rep(ii, RANK) {
					scoreV[ii].i4[i] = iround( score[i][ii] * rcFactor[ii] );
				}        
			} }
        rep(ii, RANK) {
            scoreV[ii].calculateMinAndSum(N);
        }     
		
        vector<vd> ATA(RANK, vd(RANK));
        TIME("ATA") {
            for (int i = 0; i < RANK; i += 2) {
                const LongVector<MAX_NROWS> &A = scoreV[i];
                const LongVector<MAX_NROWS> &B = (i + 1 >= RANK ? scoreV[i]: scoreV[i+1]);
                
                for (int j = 0; j <= i; j += 2) {
                    const LongVector<MAX_NROWS> &C = scoreV[j];
                    const LongVector<MAX_NROWS> &D = (j + 1 >= RANK ? scoreV[j]: scoreV[j+1]);    
					
                    FixedVector<4> tmp = scalarProd(N, A, B, C, D);
                    rep(ii, 2) rep(jj, 2) if ( i + ii < RANK && j + jj < RANK ) {
                        ATA[j+jj][i+ii] = ATA[i+ii][j+jj] = tmp[ii*2 + jj];
                    }
                }
            }
        }
        vector<vd> ATB(RANK, vd(L));
        TIME("ATB") {
            for (int i = 0; i < RANK; i += 2) {
                const LongVector<MAX_NROWS> &A = scoreV[i];
                const LongVector<MAX_NROWS> &B = (i + 1 >= RANK ? scoreV[i]: scoreV[i+1]);
                
                for (int j = 0; j < L; j += 2) {
                    const LongVector<MAX_NROWS> &C = dataV[j];
                    const LongVector<MAX_NROWS> &D = (j + 1 >= L ? dataV[j]: dataV[j+1]);    
					
                    FixedVector<4> tmp = scalarProd(N, A, B, C, D);
                    rep(ii, 2) rep(jj, 2) if ( i + ii < RANK && j + jj < L ) {
                        ATB[i+ii][j+jj] = tmp[ii*2 + jj];
                    }
                }
            }        
        }
		
        vector<vd> Q = gaussReduce(ATA, ATB);
        rep(i, 15) {
            rep(j, RANK) {        
                rep(k, 4) {                
                    const int x = i * 4 + k;
                    if ( x >= L ) continue;    
                    base[i][j][k] = Q[j][x];
                }                
            }
        }
        
        free(scoreV);
        free(dataV);
    }
} __attribute__ ((aligned(4*4))) ;

template<int RANKA, int NROWSA, int RANKB, int NROWSB>
struct ErrFunction {
    ScoreCalc<RANKA, NROWSA>* a;
    ScoreCalc<RANKB, NROWSB>* b;
    
    ErrFunction(ScoreCalc<RANKA, NROWSA>* a_, ScoreCalc<RANKB, NROWSB>* b_)
	: a(a_), b(b_) {}
    
    FVD error(const FVD &V) const {
        FixedVector<RANKA> A;
        rep(i, RANKA) A[i] = V.cAt(i);
        FixedVector<RANKB> B;
        rep(i, RANKB) B[i] = V.cAt(i + RANKA);
		
        const FixedVector<RANKA> resA = a->calculateError( A );
        const FixedVector<RANKB> resB = b->calculateError( B );
        
        FVD res;
        rep(i, RANKA) res[i] = resA.cAt(i);
        rep(i, RANKB) res[i+RANKA] = resB.cAt(i);
        return res;    
    }
    
    FVD size(const FVD &V) const {
        FixedVector<RANKA> A;
        rep(i, RANKA) A[i] = V.cAt(i);
        FixedVector<RANKB> B;
        rep(i, RANKB) B[i] = V.cAt(i + RANKA);
		
        const FixedVector<RANKA> resA = a->calculateSize( A );
        const FixedVector<RANKB> resB = b->calculateSize( B );
        
        FVD res;
        rep(i, RANKA) res[i] = resA.cAt(i);
        rep(i, RANKB) res[i+RANKA] = resB.cAt(i);
        return res;    
    }    
	/*
	 uint64_t calculateExactError(FVD V) {
	 const vd factorA( V.begin(), V.begin() + RANKA );
	 const vd factorB( V.begin() + RANKA, V.end() );    
	 
	 return a->calculateExactError(factorA) + 
	 b->calculateExactError(factorB);
	 }
	 */
    void updateBasis(FVD V) {
        const vd factorA( V.begin(), V.begin() + RANKA );
        const vd factorB( V.begin() + RANKA, V.end() );        
        a->updateBasis(factorA);
        b->updateBasis(factorB);
    }    
};

template<typename T>
struct LagrangeMinimization {
    LagrangeMinimization(const T &fun_)
	: fun(fun_), sizeIsUpdated(false), f(0.0), err(0.0) {}
    
    const T &fun;
    bool sizeIsUpdated;
    FVD f, err, size;
	
    FVD makeValidDelta(FVD &newF) {
        FVD deltaH = newF - f;
        const double minVal = 0.01;
        tr(it, deltaH) {
            if ( abs(*it) < minVal ) *it = minVal;
        }
        return deltaH;
    }    
    
    void lagrangeMinimizationStep(double aim, double stepLength, FVD &newF) {
        if ( !sizeIsUpdated) {
            size = fun.size(f);
        }
		
        FVD deltaH = makeValidDelta(newF);
        newF = f + deltaH;    
        
        FVD newErr = fun.error( newF );
        FVD newSize = fun.size( newF );
        sizeIsUpdated = true;
        
        FVD dErr = (newErr - err) / deltaH;
        const double errLengthSqr = scalarProduct( dErr, dErr );
		
        FVD dSize = (newSize - size) / deltaH;
        
        const double sumErr = sum( newErr );    
        const double s = scalarProduct(dErr, dSize);    
		
        FVD orthGradSize = dSize - dErr * (s / errLengthSqr);
        double m = *min_element( all(orthGradSize) );
        double M = *max_element( all(orthGradSize) );
        double absM = max(1.e-20, max( abs(m), abs(M) ));
        double len = -stepLength / absM;
		
        double dErrSum = sum( dErr );
        deltaH = orthGradSize * len + (aim - sumErr) / dErrSum;    
		
        err = newErr;
        size = newSize;
        f = newF;
        
        newF = f + deltaH;
        tr(it, newF) if ( *it < 1 ) *it = 1;
    }
	
    void secantStep(double aim, FVD &newF) {
        FVD deltaH = makeValidDelta(newF);
        newF = f + deltaH;    
		
        FVD newErr = fun.error( newF );
        FVD dErr = (newErr - err) / deltaH;
		
        const double sumErr = sum( newErr );
        const double dErrSum = sum( dErr );
        deltaH = (aim - sumErr) / dErrSum;    
		
        err = newErr;
        f = newF;
        sizeIsUpdated = false;
		
        newF = f + deltaH;
        tr(it, newF) if ( *it < 1 ) *it = 1;    
    }
};

vector<int> DelicatoDatCompression::decompress(const vector<int> &data) {
    cout << "Decompress start." << endl;
    Timer globTimer(" ===== Decompress total");
    
    BitInputStream bin( data );    
    const int w = bin.get<int>();
    const int h = bin.get<int>();
    const int N = w * h;
    const int L = bin.get<int>();
    const int nBad = bin.get<int>();
    cout << "decompress, nBad = " << nBad << endl;
    
    vi bitVector(N);
    if (nBad > 0) loadInts( N, bin, &bitVector.front() );    
    
    RowExtractor<NRANK_GOOD, MAX_N> *goodRowExtractor = (RowExtractor<NRANK_GOOD, MAX_N>*)memalign(16, sizeof(RowExtractor<NRANK_GOOD, MAX_N>) );
    goodRowExtractor->init(L, NRANK_GOOD);
    RowExtractor<MAX_L, MAX_NBAD> *badRowExtractor = (RowExtractor<MAX_L, MAX_NBAD>*)memalign(16, sizeof(RowExtractor<MAX_L, MAX_NBAD>) );
    badRowExtractor->init(L, MAX_L);
    
    TIME("arithmetic unpacking") {
        goodRowExtractor->load(bin, NRANK_GOOD, L, N - nBad);
        if (nBad > 0) badRowExtractor->load(bin, NRANK_BAD, L, nBad);
    }
	
    vector<int> res_V( (3 + w*h * L + 1) / 2, 0 );
    TIME("extracting rows") {
        uint16_t *resPtr = ((uint16_t*)&res_V.front());
        resPtr[0] = w;
        resPtr[1] = h;
        resPtr[2] = L;
        resPtr += 3;
		
        for (int i = 0; i < N; ++i, resPtr += L) {
            if ( unlikely(bitVector[i]) ) {
                badRowExtractor->extractNextRow(NRANK_BAD, L, resPtr);
            }
            else {    
                goodRowExtractor->extractNextRow(NRANK_GOOD, L, resPtr);            
            }
        }
    }
    
    free(goodRowExtractor);
    free(badRowExtractor);
    
    return res_V;
}

// int DelicatoDatCompression::GetCompressionType() { return TraceCompressor::Delicato; }
int DelicatoDatCompression::GetCompressionType() { return 1;}

vector<int> DelicatoDatCompression::compress(const vector<int> &data) {
    Timer globTimer(" ===== Compress total");
    cout << "Compress start." << endl;
    
    const uint16_t *const dataPtr = (uint16_t*)&data.front();
    
    const int w = dataPtr[0];
    const int h = dataPtr[1];
    const int L = dataPtr[2];
    const int N = h * w;
    
    cout << "w = " << w << ", h = " << h << ", L = " << L << ", N = " << N << endl;
	
    TransposeSum ts(L);
    TIME("Transpose matrix") {
        const double q = double(N - NSAMPLING_INNER - 5) / double(NSAMPLING_OUTER - 1);
        rep( i, NSAMPLING_OUTER ) {
            const int k = iround( q * i );
            const uint16_t *rowPtr = dataPtr + 3 + k * L;
            
            rep(j, NSAMPLING_INNER ) {
                ts.pushRow( rowPtr );
                rowPtr += L;
            }
        }
    }
    const vector<vf> eigenVectors = ts.calculateEigenVectors();
	
    ScoreCalc<NRANK_GOOD, MAX_N> *sc = (ScoreCalc<NRANK_GOOD, MAX_N>*)memalign(16, sizeof(ScoreCalc<NRANK_GOOD, MAX_N>) );
    
    sc->init( eigenVectors, dataPtr );
    TIME("Scores") {
        const uint16_t *ptr = dataPtr + 3;
        rep(i, N) {
            sc->pushRow( ptr );
            ptr += L;
        }
    }    
    
    vi badIdx = sc->getBadIdx();
    const int nBad = badIdx.size();
    const int nGood = N - nBad;
    badIdx.push_back( -1 ); // sentry
    cout << "nBad: " << nBad << endl; 
	
    ScoreCalc<NRANK_GOOD, MAX_N> *gsc = (ScoreCalc<NRANK_GOOD, MAX_N>*)memalign(16, sizeof(ScoreCalc<NRANK_GOOD, MAX_N>) );
    gsc->init( eigenVectors, dataPtr );
    gsc->N = nGood;    
    vi goodIdx(nGood);
    TIME("Good idx") {
        int i = 0, k = 0;
        rep(j, N) {
            if ( unlikely(j == badIdx[i]) ) ++i;
            else goodIdx[k++] = j;
        }
    }
	
    TIME("Copying data") {    
        int j = 0, k = 0;
        rep(i, N) {
            if ( likely( badIdx[j] != i ) ) {
                gsc->length[k] = sc->length[i];
                rep(j, NRANK_GOOD/4) {
                    gsc->scoreV4[k][j] = sc->scoreV4[i][j];
                }                
                ++k;
            }
            else ++j;
        }
    }
    free(sc);
	
    ScoreCalc<NRANK_BAD, MAX_NBAD> *bsc = (ScoreCalc<NRANK_BAD, MAX_NBAD>*)memalign(16, sizeof(ScoreCalc<NRANK_BAD, MAX_NBAD>) );
    bsc->init( eigenVectors, dataPtr );
    
    TIME("Scores bad") {
        rep(i, nBad) {
            bsc->pushRow( dataPtr + 3 + L * badIdx[i] );
        }
    }    
    
    const double orthErrGood = gsc->calcOrthogonalErrorSum();
    const double orthErrBad = bsc->calcOrthogonalErrorSum();
    const double orthErr = orthErrGood + orthErrBad;
	
    TIME("min/max gsc/bsc") {
        gsc->calculateMinAndMax();
        bsc->calculateMinAndMax();
    }
	
    const double aim1 = AIM1 * N * L - orthErr;
    cout << "aim = " << aim1 / (N * L) << endl;    
    cout << "orthErr from good = " << orthErrGood / (N * L)  << endl;
    cout << "ortErr from bad   = " << orthErrBad / (N * L) << endl;    
	
    assert( aim1 > 0 );
    ErrFunction<NRANK_BAD, MAX_NBAD, NRANK_GOOD, MAX_N> errF(bsc, gsc);
    
    LagrangeMinimization<typeof(errF)> laMi(errF);
    FVD factorV( 80.0 );
    TIME("Secant 1") {
        repeat(2) laMi.secantStep(aim1, factorV);
    }
	
    TIME("Lagrange 1") {
        repeat(4) laMi.lagrangeMinimizationStep(aim1, LAGRANGE_STEP_SIZE, factorV);
        cout << "lagrange done" << endl;
        repeat(2) laMi.secantStep(aim1, factorV);
    }
    const double estimatedError1 = sum( errF.error(factorV) );
    cout << "estimatedError1 " << estimatedError1 / (N * L) << endl;
    
    const vd factorBad( factorV.begin(), factorV.begin() + NRANK_BAD );
    const vd factorGood( factorV.begin() + NRANK_BAD, factorV.end() );        
    
    gsc->updateToOptimalBasis( factorGood, goodIdx );
    bsc->updateToOptimalBasis( factorBad, badIdx );
    uint64_t dfs1 = gsc->calculateExactError(factorGood, goodIdx);
    uint64_t dfs2 = 	bsc->calculateExactError(factorBad, badIdx);   
    // const uint64_t diffSum1 = gsc->calculateExactError(factorGood, goodIdx) + 
    //     bsc->calculateExactError(factorBad, badIdx);    
    const uint64_t diffSum1 = dfs1 + dfs2;
    
    const double error1 = double(diffSum1) / (N * L);
    cout << "Exact error first try: " << error1 << endl;
    if ( nBad > 45000 ) {    // lack of time
        cout << "nBad large, bailing" << endl;
        
        if (error1 < ACCEPT2 || error1 > 36.0) {
            const double aim3 = (estimatedError1 - diffSum1) + AIM_BAIL * N * L;
            FVD oldFactorV = factorV;
            TIME("Secant 4") {
                repeat(2) laMi.secantStep(aim3, factorV);
            }
            errF.updateBasis( factorV / oldFactorV );        
        }
    }
    else if ( error1 < ACCEPT1 || error1 > 36.0 ) {
        const double aim2 = (estimatedError1 - diffSum1) + (error1 < ACCEPT1 ? AIM2_HIGH: AIM2_LOW) * N * L;
        assert( aim2 > 0 );
        FVD oldFactorV = factorV;
        
        TIME("Lagrange 2") {
            repeat(4) laMi.lagrangeMinimizationStep(aim2, LAGRANGE_STEP_SIZE, factorV);
            repeat(2) laMi.secantStep(aim2, factorV);
        }            
        const double estimatedError2 = sum( errF.error(factorV) );
        
        errF.updateBasis( factorV / oldFactorV );
        const vd factorBad( factorV.begin(), factorV.begin() + NRANK_BAD );
        const vd factorGood( factorV.begin() + NRANK_BAD, factorV.end() );    
        const uint64_t diffSum2 = gsc->calculateExactError(factorGood, goodIdx) + 
		bsc->calculateExactError(factorBad, badIdx);            
		
        const double error2 = double(diffSum2) / (N * L);
        cout << "Exact error second try: " << error2 << endl;
		
        if ( error2 < ACCEPT2 || error2 > 36.0 ) {
            const double aim3 = (estimatedError2 - diffSum2) + AIM3 * N * L;
            FVD oldFactorV = factorV;
            TIME("Secant 4") {
                repeat(2) laMi.secantStep(aim3, factorV);
            }
            errF.updateBasis( factorV / oldFactorV );
        }
    }
    
    BitOutputStream bout;
    TIME("Compressing") {
        bout.put<int>(w);
        bout.put<int>(h);
        bout.put<int>(L);
        bout.put<int>(nBad);
        
        vi bitVectorOrg(N, 0);
        rep(i, nBad) {
            bitVectorOrg[ badIdx[i] ] = 1;
        }
        if (nBad > 0) storeInts(N, &bitVectorOrg.front(), bout );    
		
        const vd factorBad( factorV.begin(), factorV.begin() + NRANK_BAD );
        const vd factorGood( factorV.begin() + NRANK_BAD, factorV.end() );                
        
        gsc->saveBasis(bout);
        gsc->saveScores(bout, factorGood);
        if ( nBad > 0 ) {
            bsc->saveBasis(bout);
            bsc->saveScores(bout, factorBad);
        }
    }    
    
    free(bsc);
    free(gsc);
    return bout.data();
}
