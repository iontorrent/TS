NCC = nvcc
CC = g++

#VECT_OPTS = -msse3 -fabi-version=4

BIN=xtalk_sim

#CUDA
CUDA_DIR=/usr/local/cuda
CUDA_INCLUDE= -I$(CUDA_DIR)/include
CUDA_LIBRARY=-L$(CUDA_DIR)/lib64 -L$(CUDA_DIR)/lib -lcuda -lcudart
#CUDA_EXTRA_FLAGS=  -O3  --ptxas-options=-v  -arch sm_20 -use_fast_math 
CUDA_EXTRA_FLAGS=  -O3 -arch sm_35 --ptxas-options=-v -use_fast_math 

#C/CPP
CFLAGS= -O3  
CLIBS= -lpthread
CINC= 


OBJ =  \
  xtalk_sim.o    \
  DiffEqModel.o  \
  WorkerInfoQueue.o \
  DiffEqModel_Init.o \
  utils.o            \
  sim_incorp_flux.o \
	DelsqCUDA.o


all: $(OBJ)
	$(CC) $(OBJ) -o $(BIN) $(CLIBS) $(CUDA_LIBRARY)

clean:
	rm -f $(BIN) *.o 

.SUFFIXES: .c .cpp .cu

.cpp.o:
	$(CC) $(DEFINES) $(VECT_OPTS) $(CFLAGS) $(CINC) $(CUDA_INCLUDE) -c $<

.c.o:
	$(CC) $(DEFINES) $(VECT_OPTS) $(CFLAGS) $(CINC) $(CUDA_INCLUDE) -c $<

.cu.o:
	$(NCC) -c $(DEFINES)  $(CUDA_EXTRA_FLAGS) $(CUDA_INCLUDE) $(CINC) $<

# xtalk_sim: xtalk_sim.cpp DiffEqModel.cpp WorkerInfoQueue.cpp DiffEqModel_Init.cpp utils.cpp xtalk_sim.h DiffEqModel.h WorkerInfoQueue.h SignalAverager.h
#	g++ -o xtalk_sim -O3 $(VECT_OPTS) $(CUDA_INC) xtalk_sim.cpp DiffEqModel.cpp WorkerInfoQueue.cpp DiffEqModel_Init.cpp utils.cpp -lpthread $(CUDA_LIB) 


