QT += widgets core gui printsupport

CONFIG += c++11
INCLUDEPATH += /usr/include/hdf5/serial/ ../ ../file-io/ ../Util/ ../Mask/ ../AnalysisOrg ../TsInputUtil/ ../Wells/ ../Image/ ../BaseCaller/ ../../build/bamtools-2.4.0.20150702+git15eadb925f-install/include/bamtools/ ../VariantCaller/Bookkeeping/ ../../external/jsoncpp-src-amalgated0.6.0-rc1/ ../../build/armadillo-6.100.0+ion1/include/

# for 18.04
    INCLUDEPATH += /usr/include/bamtools/
# for 14.04
#    INCLUDEPATH += ../../build/bamtools-2.4.0.20150702+git15eadb925f-install/include/bamtools/


QMAKE_CXXFLAGS += -mavx -DDATAVIEWER

HEADERS       = \
    dialog.h \
    AlignmentTab.h \
    AlignmentSpatial.h \
    BfMaskTab.h \
    RawTab.h \
    qcustomplot.h \
    SpatialPlot.h \
    RawSpatial.h \
    WellsTab.h \
    WellsSpatial.h \
    BfMaskSpatial.h \
    NoiseTab.h \
    NoiseSpatial.h \
    GainSpatial.h \
    GainTab.h \
    modeltab.h \
    MicroscopeSpatial.h \
    MicroscopeTab.h \
    NumpySpatial.h \
    NumpyTab.h


SOURCES       = main.cpp \
    dialog.cpp \
    AlignmentTab.cpp \
    BfMaskTab.cpp \
    RawTab.cpp \
    qcustomplot.cpp \
    SpatialPlot.cpp \
    RawSpatial.cpp \
    WellsTab.cpp \
    WellsSpatial.cpp \
    MicroscopeTab.cpp \
    MicroscopeSpatial.cpp \
    ../Image/CorrNoiseCorrector.cpp \
    ../Image/ComparatorNoiseCorrector.cpp \
    ../Image/AdvCompr.cpp \
    ../Image/PCACompression.cpp \
    ../Image/deInterlace.cpp \
    ../Image/ChipIdDecoder.cpp \
    ../Wells/RawWells.cpp \
    ../Wells/RawWellsV1.cpp \
    ../Util/IonErr.cpp \
    ../Util/Utils.cpp \
    ../Util/IonH5File.cpp \
    ../LinuxCompat.cpp \
    ../TsInputUtil/IonVersion.cpp \
    BfMaskSpatial.cpp \
    NoiseTab.cpp \
    NoiseSpatial.cpp \
    GainSpatial.cpp \
    GainTab.cpp \
    AlignmentSpatial.cpp \
    modeltab.cpp \
    ../Wells/WellsNormalization.cpp \
    ../Mask/Mask.cpp \
    ../BaseCaller/BaseCallerParameters.cpp \
    ../BaseCaller/ReadClassMap.cpp \
#    ../BaseCaller/OrderedDatasetWriter.cpp \
#    ../Calibration/FlowAlignment.cpp \
#    ../BaseCaller/PhaseEstimator.cpp \
#    ../BaseCaller/PerBaseQual.cpp \
#    ../BaseCaller/DpTreePhaser.cpp \
#    ../BaseCaller/PIDloop.cpp \
    ../Util/OptArgs.cpp \
    ../file-io/ion_util.c \
    ../file-io/ion_error.c \
#    ../BaseCaller/BaseCallerFilters.cpp \
    ../../external/jsoncpp-src-amalgated0.6.0-rc1/jsoncpp.cpp \
    ../ionstats/ionstats_alignment.cpp \
    ../ionstats/ionstats_alignment_summary.cpp \
    ../ionstats/ionstats_data.cpp \
    NumpySpatial.cpp \
    NumpyTab.cpp

message($$QMAKE_HOST.version)

unix:!mac:!vxworks:!integrity:!haiku:LIBS += -lm  -lz

# for 18.04
    unix:!mac:!vxworks:!integrity:!haiku:LIBS += -lhdf5_serial -lbamtools 
# for 14.04
#    unix:!mac:!vxworks:!integrity:!haiku:LIBS +=  "-lhdf5 ../../build/bamtools-2.4.0.20150702+git15eadb925f-install/lib/bamtools/libbamtools.a"


# install
#target.path = DataViewer
#INSTALLS += target
