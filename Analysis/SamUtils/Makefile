CC=g++
ANALYSIS=../
#samtools library must be made already, run make lib in the samtools directory to get it
SAMTOOLS=../../external/samtools-0.1.18/
CFLAGS=-g -Wall -Werror -DALIGNSTATS_IGNORE
LDFLAGS=-L. -L$(SAMTOOLS) -L$(SAMTOOLS)/bcftools/ -L./types/ -lpthread -lbam -lz
EXE_FILES = alignStats
.PHONY: all clean

all: $(EXE_FILES)

clean:
	-rm $(EXE_FILES) *.o types/*.o
	-rm $(ANALYSIS)/OptArgs.o
	-rm $(ANALYSIS)/Utils.o

alignStats: Cigar.o Sequence.o Qual.o MD.o BAMRead.o BAMUtils.o BAMReader.o alignStats_driver.o Utils.o OptArgs.o
	$(CC) $(CFLAGS) -o alignStats types/*.o alignStats_driver.o alignStats.o BAMUtils.o BAMReader.o $(ANALYSIS)/OptArgs.o $(ANALYSIS)/Utils.o $(ANALYSIS)/IonErr.o $(LDFLAGS)
	
alignStats_driver.o: BAMReader.o alignStats.o 
	$(CC) $(CFLAGS) -c alignStats_driver.cpp -I $(SAMTOOLS) -I $(ANALYSIS) -o alignStats_driver.o

alignStats.o: IonErr.o BAMReader.o BAMUtils.o alignStats.cpp alignStats.h 
	$(CC) $(CFLAGS) -c alignStats.cpp -I types/ -I $(SAMTOOLS) -I $(ANALYSIS) -I ./ -o alignStats.o
	

BAMUtils.o: BAMUtils.cpp BAMUtils.h BAMRead.o
	$(CC) $(CFLAGS) -c BAMUtils.cpp -I types/ -I $(SAMTOOLS) -I $(ANALYSIS) -I ./ -o BAMUtils.o
	
BAMReader.o: BAMReader.cpp BAMReader.h types/BAMRead.cpp types/BAMRead.h
	$(CC) $(CFLAGS) -c BAMReader.cpp -I $(SAMTOOLS) -I types/ -I $(ANALYSIS) -I ./ -o BAMReader.o

BAMRead.o: types/BAMRead.cpp types/BAMRead.h types/samutils_types.h types/Cigar.o types/Qual.o types/Sequence.o types/MD.o
	$(CC) $(CFLAGS) -c types/BAMRead.cpp -I $(SAMTOOLS) -I ./ -o types/BAMRead.o
	
	
Cigar.o: types/Cigar.cpp types/Cigar.h
	$(CC) $(CFLAGS) -c types/Cigar.cpp -I types/ -I $(SAMTOOLS) -o types/Cigar.o
	
Qual.o: types/Qual.cpp types/Qual.h
	$(CC) $(CFLAGS) -c types/Qual.cpp -I types/ -I $(SAMTOOLS) -o types/Qual.o
	
Sequence.o: types/Sequence.cpp types/Sequence.h
	$(CC) $(CFLAGS) -c types/Sequence.cpp -I types/ -I $(SAMTOOLS) -o types/Sequence.o
	
MD.o: types/MD.cpp types/MD.h
	$(CC) $(CFLAGS) -c types/MD.cpp -I types/ -I $(SAMTOOLS) -o types/MD.o
	
Utils.o: $(ANALYSIS)/Utils.cpp $(ANALYSIS)/Utils.h
	$(CC) $(CFLAGS) -c $(ANALYSIS)/Utils.cpp -I $(ANALYSIS) -o $(ANALYSIS)/Utils.o
	
OptArgs.o: $(ANALYSIS)/OptArgs.cpp $(ANALYSIS)/OptArgs.h
	$(CC) $(CFLAGS) -c $(ANALYSIS)/OptArgs.cpp -I $(ANALYSIS) -o $(ANALYSIS)/OptArgs.o

IonErr.o: $(ANALYSIS)/IonErr.cpp $(ANALYSIS)/IonErr.h
	$(CC) $(CFLAGS) -c $(ANALYSIS)/IonErr.cpp -I $(ANALYSIS) -o $(ANALYSIS)/IonErr.o

