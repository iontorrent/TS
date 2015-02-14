/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#include "realign_proxy.h"
#include "realign_wrapper_imp.h"
#include "realign_wrapper_context_imp.h"

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
RealignProxy* createContextAligner ()
{
    return new ContAlignImp;
}
