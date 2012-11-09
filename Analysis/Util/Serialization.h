/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <cassert>
#include <fstream>
#include <string>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

template <class T>
void SerializeIn(const std::string& dirName, const std::string& fileName, T& x)
{
  std::string   filePath = dirName + std::string("/") + fileName;
  std::ifstream inStream(filePath.c_str());
  assert(inStream);

  boost::archive::text_iarchive inArchive(inStream);
  inArchive & x;
}

template <class T>
void SerializeOut(const std::string& dirName, const std::string& fileName, const T& x)
{
  std::string   filePath = dirName + std::string("/") + fileName;
  std::ofstream outStream(filePath.c_str());
  assert(outStream);

  boost::archive::text_oarchive outArchive(outStream);
  outArchive & x;
}

#endif // SERIALIZATION_H
