QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        BSplineBasisFunction.C \
        main.cpp

# Eigen3
INCLUDEPATH += /home/pascal/eigen

# OpenCV
INCLUDEPATH += /usr/local/include \
               /usr/local/include/opencv \
               /usr/local/include/opencv2

LIBS += /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_core.so    \
        /usr/local/lib/libopencv_imgproc.so \
        /usr/local/lib/libopencv_imgcodecs.so

LIBS+=/usr/local/lib/libopencv_shape.so
LIBS+=/usr/local/lib/libopencv_videoio.so

# OpenNI
INCLUDEPATH += /home/pascal/OpenNI-Linux-x64-2.3.0.63/Include

LIBS += /home/pascal/OpenNI-Linux-x64-2.3.0.63/Tools/libOpenNI2.so  \
        /usr/lib/x86_64-linux-gnu/libglut.so


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    BSplineBasisFunction.h \
    OniSampleUtilities.h
