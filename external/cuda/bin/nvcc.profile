
TOP              = $(_HERE_)/..

LD_LIBRARY_PATH += $(TOP)/lib:$(TOP)/extools/lib:
PATH            += $(TOP)/open64/bin:$(_HERE_):

INCLUDES        +=  "-I$(TOP)/include" "-I$(TOP)/include/cudart" $(_SPACE_)

LIBRARIES        =+ $(_SPACE_) "-L$(TOP)/lib$(_TARGET_SIZE_)" -lcudart

CUDAFE_FLAGS    +=
OPENCC_FLAGS    +=
PTXAS_FLAGS     +=
