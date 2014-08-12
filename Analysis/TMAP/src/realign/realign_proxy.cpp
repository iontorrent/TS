/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include "realign_proxy.h"
#include "realign_wrapper_imp.h"

RealignProxy::~RealignProxy ()
{
}

RealignProxy* createRealigner ()
{
    return new RealignImp;
}
RealignProxy* createRealigner (unsigned reserve_size, unsigned clipping_size)
{
    return new RealignImp (reserve_size, clipping_size);
}
