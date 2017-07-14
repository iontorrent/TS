/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */

#include "dialog.h"

#include <QApplication>

#include <QDialog>
#include <QtWidgets>


//! [0]
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    Dialog dialog;

    dialog.show();

    return app.exec();
}
//! [0]
