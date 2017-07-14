QT += widgets core gui printsupport

CONFIG += c++11
INCLUDEPATH += /usr/include/hdf5/serial/ ../ ../Util/ ../Mask/ ../TsInputUtil/ ../Wells/ ../Image/ ../../build/bamtools-2.4.0.20150702+git15eadb925f-install/include/bamtools/ ../VariantCaller/Bookkeeping/

QMAKE_CXXFLAGS += -mavx

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
    modeltab.h


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
    ../Calibration/FlowAlignment.cpp \
    modeltab.cpp


unix:!mac:!vxworks:!integrity:!haiku:LIBS += -lm -lhdf5_serial "../../build/bamtools-2.4.0.20150702+git15eadb925f-install/lib/bamtools/libbamtools.a" -lz

# install
#target.path = DataViewer
#INSTALLS += target
