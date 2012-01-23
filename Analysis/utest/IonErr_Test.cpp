/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <gtest/gtest.h>
#include "IonErr.h"
#include <stdexcept>
using namespace std;

TEST(IonErr_Test, ThrowTest) {

  ASSERT_EQ(IonErr::GetThrowStatus(), false);
  IonErr::SetThrowStatus(true);
  ASSERT_EQ(IonErr::GetThrowStatus(), true);
  ASSERT_THROW( {ION_ABORT("boom")}, runtime_error);

  IonErr::SetThrowStatus(false);
  ASSERT_EQ(IonErr::GetThrowStatus(), false);
  ASSERT_DEATH(ION_ABORT("boom"), ".*boom.*");
}
