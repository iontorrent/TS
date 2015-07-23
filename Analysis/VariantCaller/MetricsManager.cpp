/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     MetricsManager.cpp
//! @ingroup  VariantCaller
//! @brief    Collects analysis metrics and saves them to a json file


#include "MetricsManager.h"

#include <fstream>
#include "json/json.h"

MetricsAccumulator& MetricsManager::NewAccumulator()
{
  accumulators_.push_back(MetricsAccumulator());
  return accumulators_.back();
}



void MetricsAccumulator::CollectMetrics(list<PositionInProgress>::iterator& position_ticket, int haplotype_length,  const ReferenceReader* ref_reader)
{

  long next_pos = min(position_ticket->pos + haplotype_length, position_ticket->target_end);
  ReferenceReader::iterator ref_ptr  = ref_reader->iter(position_ticket->chr, position_ticket->pos);

  // Iterate over positions covered by this call to GenerateCandidates
  for (long pos = position_ticket->pos; pos < next_pos; ++pos, ++ref_ptr) {

    int forward_map[8], reverse_map[8];
    for (int idx = 0; idx < 8; ++idx)
      forward_map[idx] = reverse_map[idx] = 0;
    int forward_total = 0;
    int reverse_total = 0;

    const Alignment * __restrict__ rai_end = position_ticket->end;
    for (const Alignment * __restrict__ rai = position_ticket->begin; rai != rai_end; rai = rai->next) {
      if (rai->filtered)
        continue;
      if (pos < rai->start or pos >= rai->end)
        continue;

      int read_pos = pos - rai->alignment.Position;
      if (rai->refmap_code[read_pos] != 'X' and rai->refmap_code[read_pos] != 'M')    // match or substitution
        continue;

      char read_base = *(rai->refmap_start[read_pos]);

      if (rai->alignment.IsReverseStrand()) {
        reverse_total++;
        reverse_map[read_base&7]++;
      } else {
        forward_total++;
        forward_map[read_base&7]++;
      }
    }

    if (forward_total < 30 or reverse_total < 30)
      continue;

    char ref_base = *ref_ptr;
    char bases[5] = "ACGT";
    for (const char *read_base = bases; read_base < &bases[4]; ++read_base) {
      if (forward_map[(*read_base)&7] > (forward_total / 1000) and
          forward_map[(*read_base)&7] < (15*forward_total / 100) and
          reverse_map[(*read_base)&7] > (reverse_total / 1000) and
          reverse_map[(*read_base)&7] < (15*reverse_total / 100)) {
        substitution_events[(ref_base&7) + (((*read_base)&7)<<3)] += forward_map[(*read_base)&7] + reverse_map[(*read_base)&7];
      }
    }
  }
}




void MetricsManager::FinalizeAndSave(const string& output_json)
{
  MetricsAccumulator final;

  for (list<MetricsAccumulator>::iterator I = accumulators_.begin(); I != accumulators_.end(); ++I)
    final += *I;

  Json::Value json(Json::objectValue);

  long int sum = 0;
  char bases[5] = "ACGT";
  for(unsigned i = 0; i < 4; ++i)
    for(unsigned j = 0; j < 4; ++j)
      if(i != j)
        sum += final.substitution_events[(bases[i]&7) + ((bases[j]&7)<<3)];

  if (sum > 0)
    json["metrics"]["deamination_metric"] = (final.substitution_events[('C'&7) + (('T'&7)<<3)] +
        final.substitution_events[('G'&7) + (('A'&7)<<3)]) / (double)sum;
  else
    json["metrics"]["deamination_metric"] = 0;

  json["metrics"]["A>C"] = (Json::Int64)final.substitution_events[('A'&7) + (('C'&7)<<3)];
  json["metrics"]["A>G"] = (Json::Int64)final.substitution_events[('A'&7) + (('G'&7)<<3)];
  json["metrics"]["A>T"] = (Json::Int64)final.substitution_events[('A'&7) + (('T'&7)<<3)];
  json["metrics"]["C>A"] = (Json::Int64)final.substitution_events[('C'&7) + (('A'&7)<<3)];
  json["metrics"]["C>G"] = (Json::Int64)final.substitution_events[('C'&7) + (('G'&7)<<3)];
  json["metrics"]["C>T"] = (Json::Int64)final.substitution_events[('C'&7) + (('T'&7)<<3)];
  json["metrics"]["G>A"] = (Json::Int64)final.substitution_events[('G'&7) + (('A'&7)<<3)];
  json["metrics"]["G>C"] = (Json::Int64)final.substitution_events[('G'&7) + (('C'&7)<<3)];
  json["metrics"]["G>T"] = (Json::Int64)final.substitution_events[('G'&7) + (('T'&7)<<3)];
  json["metrics"]["T>A"] = (Json::Int64)final.substitution_events[('T'&7) + (('A'&7)<<3)];
  json["metrics"]["T>C"] = (Json::Int64)final.substitution_events[('T'&7) + (('C'&7)<<3)];
  json["metrics"]["T>G"] = (Json::Int64)final.substitution_events[('T'&7) + (('G'&7)<<3)];

  ofstream out(output_json.c_str(), ios::out);
  if (out.good())
    out << json.toStyledString();
  else
    cerr << "WARNING: Unable to write JSON file " << output_json << endl;

}



