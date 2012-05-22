CC=clang++
ANALYSIS=../Util/
#samtools library must be made already, run make lib in the samtools directory to get it
SAMTOOLS=../../external/samtools-0.1.18/
CFLAGS=-g -Wall -DALIGNSTATS_IGNORE
LDFLAGS=-L. -L$(SAMTOOLS) -L$(SAMTOOLS)/bcftools/ -L./types/ -lpthread -lbam -lz
EXE_FILES = alignStats SamVsSam
EXAMPLE_FILES = example
.PHONY: all clean

all: $(EXE_FILES)

clean:
	-rm $(EXE_FILES) *.o types/*.o
	-rm $(ANALYSIS)/*.o

examples: $(EXAMPLE_FILES)
	$(CC) $(CFLAGS) -o example example.o types/*.o BAMReader.o BAMUtils.o $(ANALYSIS)/Utils.o $(LDFLAGS)

example: types/Cigar.o types/Sequence.o types/BAMRead.o types/MD.o BAMUtils.o BAMReader.o example.o
	$(CC) $(CFLAGS) -o example types/*.o example.o BAMUtils.o BAMReader.o $(LDFLAGS) $(ANALYSIS)/Utils.o

SamVsSam: types/Cigar.o types/Sequence.o types/BAMRead.o types/MD.o BAMUtils.o BAMReader.o SamVsSam.o
	$(CC) $(CFLAGS) -o SamVsSam types/*.o SamVsSam.o BAMUtils.o BAMReader.o $(LDFLAGS) $(ANALYSIS)/Utils.o

alignStats: types/Cigar.o types/Sequence.o types/Qual.o types/MD.o types/BAMRead.o BAMUtils.o BAMReader.o alignStats_driver.o $(ANALYSIS)/Utils.o $(ANALYSIS)/OptArgs.o $(ANALYSIS)/IonErr.o
	$(CC) $(CFLAGS) -o alignStats types/*.o alignStats_driver.o alignStats.o BAMUtils.o BAMReader.o $(ANALYSIS)/OptArgs.o $(ANALYSIS)/Utils.o $(ANALYSIS)/IonErr.o $(LDFLAGS)

example.o: example.cpp
	$(CC) $(CFLAGS) -c example.cpp -I $(SAMTOOLS) -I $(ANALYSIS) -o example.o

SamVsSam.o: SamVsSam.cpp
	$(CC) $(CFLAGS) -c SamVsSam.cpp -I $(SAMTOOLS) -I $(ANALYSIS) -o SamVsSam.o

alignStats_driver.o: BAMReader.o alignStats.o 
	$(CC) $(CFLAGS) -c alignStats_driver.cpp -I $(SAMTOOLS) -I $(ANALYSIS) -o alignStats_driver.o

alignStats.o: $(ANALYSIS)/IonErr.o BAMReader.o BAMUtils.o alignStats.cpp alignStats.h 
	$(CC) $(CFLAGS) -c alignStats.cpp -I types/ -I $(SAMTOOLS) -I $(ANALYSIS) -I ./ -o alignStats.o


BAMUtils.o: BAMUtils.cpp BAMUtils.h types/BAMRead.o
	$(CC) $(CFLAGS) -c BAMUtils.cpp -I types/ -I $(SAMTOOLS) -I $(ANALYSIS) -I ./ -o BAMUtils.o

BAMReader.o: BAMReader.cpp BAMReader.h types/BAMRead.cpp types/BAMRead.h
	$(CC) $(CFLAGS) -c BAMReader.cpp -I $(SAMTOOLS) -I types/ -I $(ANALYSIS) -I ./ -o BAMReader.o

types/BAMRead.o: types/BAMRead.cpp types/BAMRead.h types/samutils_types.h types/Cigar.o types/Qual.o types/Sequence.o types/MD.o
	$(CC) $(CFLAGS) -c types/BAMRead.cpp -I $(SAMTOOLS) -I ./ -o types/BAMRead.o

types/Cigar.o: types/Cigar.cpp types/Cigar.h
	$(CC) $(CFLAGS) -c types/Cigar.cpp -I types/ -I $(SAMTOOLS) -o types/Cigar.o

types/Qual.o: types/Qual.cpp types/Qual.h
	$(CC) $(CFLAGS) -c types/Qual.cpp -I types/ -I $(SAMTOOLS) -o types/Qual.o

types/Sequence.o: types/Sequence.cpp types/Sequence.h
	$(CC) $(CFLAGS) -c types/Sequence.cpp -I types/ -I $(SAMTOOLS) -o types/Sequence.o

types/MD.o: types/MD.cpp types/MD.h
	$(CC) $(CFLAGS) -c types/MD.cpp -I types/ -I $(SAMTOOLS) -o types/MD.o

$(ANALYSIS)/Utils.o: $(ANALYSIS)/Utils.cpp $(ANALYSIS)/Utils.h
	$(CC) $(CFLAGS) -c $(ANALYSIS)/Utils.cpp -I $(ANALYSIS) -o $(ANALYSIS)/Utils.o

$(ANALYSIS)/OptArgs.o: $(ANALYSIS)/OptArgs.cpp $(ANALYSIS)/OptArgs.h
	$(CC) $(CFLAGS) -c $(ANALYSIS)/OptArgs.cpp -I $(ANALYSIS) -o $(ANALYSIS)/OptArgs.o

$(ANALYSIS)/IonErr.o: $(ANALYSIS)/IonErr.cpp $(ANALYSIS)/IonErr.h
	$(CC) $(CFLAGS) -c $(ANALYSIS)/IonErr.cpp -I $(ANALYSIS) -o $(ANALYSIS)/IonErr.o

