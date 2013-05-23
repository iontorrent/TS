TEMPLATE = app
CONFIG += console
CONFIG -= qt

INCLUDEPATH += \
    ../../external/jsoncpp-src-amalgated0.6.0-rc1

SOURCES += \
    logdigger.cpp \
    ../../external/jsoncpp-src-amalgated0.6.0-rc1/jsoncpp.cpp

LIBS += \
    /usr/lib/libboost_filesystem.so

