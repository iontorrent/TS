
When ionstats alignment is called with the --output-h5 option it will produce a
data file with summary alignment error metrics.  This document describes the
format and contents of the file.

The file will by default will be named ionstats_error_summary.h5, written in 
HDF5 format.  The exact contents can vary according to the usage of some
options in the call to ionstats alignment.  The contents can be broken down
into three types of data structures - ErrorData, HpData and RegionalSummary.
An important feature of all of these data structures is that they can be
easily aggregated across different datasets - in particular there is a general
preference to storing integer-valued numerators & denominators separately,
as opposed to ever storing floating-point ratios.

## 
## 1 - ErrorData structure:
## 

The ErrorData structure provides data about per-base or per-flow errors
It is implemented as an HDF5 group with four datasets:

  region_origin    Zero-based x,y region origin                      (uint32)
  region_dim       Width and height of region                        (uint32)
  error_data_dim   Number of rows & columns in the error_data matrix (uint32)
  error_data       The error data, as described below                (uint64)

The error_data matrix consists of 6 columns and a row for each position, which
can be a flow number or a base position.  The 6 columns are:

  ins         - the number of insertions at the position
  del         - the number of insertions at the position
  sub         - the number of insertions at the position
  align_start - the number of alignments starting at the position
  align_stop  - the number of alignments stopping at the position
  depth       - depth at this position

The depth field warrants a little explanation.  In the case of per-base data it
is simply the read depth at the position, it could also be computed from the
align_start and align_stop data and so it is redundant.  But in the case of per
flow data it reports the number of incorporating flows at the flow position, 
derived from the read sequence and not the reference.  For per-flow data the
total number of alignments spanning the position can still be derived from the
align_start and align_stop values.

The following ErrorData structures may be provided in the HDF5 file:

  per_base/error_data - overall per-base data
  per_flow/error_data - overall per-flow data
  per_read_group/<barcode>/per_base/error_data - per-barcode per-base data
  per_read_group/<barcode>/per_flow/error_data - per-barcode per-flow data

The option --evaluate-flow in the call to ionstats alignment will control
whether or not per-flow data are reported.


## 
## 2 - HpData structure:
## 

The HpData structure provides information about per-homopolymer errors.  It
is implemented as an HDF5 group with seven datasets:

  region_origin    Zero-based x,y region origin                      (uint32)
  region_dim       Width and height of region                        (uint32)
  hp_data_dim      Number of rows & columns in the hp_data matrix    (uint32)
  A                A data, as described below                        (uint64)
  C                C data, as described below                        (uint64)
  G                G data, as described below                        (uint64)
  T                T data, as described below                        (uint64)

The hp matrices have a row for each reference HP length and a column for each 
observed HP length

The following HpData structures may be provided in the HDF5 file:

  per_base/per_hp - overall per-hp data
  per_read_group/<barcode>/per_hp - per-barcode per-hp data

The option --evaluate-hp in the call to ionstats alignment will control
whether or not the per-hp data are reported.  The --max-hp option specifies
the max HP length to consider.


## 
## 3 - RegionalSummary structure:
## 

The RegionalSummary structure provides regional information about per-flow and
per-hp errors.  It is implemented as an HDF5 group with four datasets:

  region_origin    Zero-based x,y region origin                      (uint32)
  region_dim       Width and height of region                        (uint32)
  n_err            Number of base errors in the region               (uint64)
  n_aligned        Number of bases aligned in the region             (uint64)
  data_dim         Nubmer of flows & max HP length                   (uint32)
  hp_count         The number of HPs of each length in each flow     (uint64)
  hp_err           # erroneous HPs of each length in each flow       (uint64)

The hp_count and hp_err matrices have dimensions as specified in data_dim
with entries providing the overall number (hp_count) and erroneous number
(hp_err) of each homopolymer length in each flow for the region.

The following HpData structures may be provided in the HDF5 file:

  per_region/<regions>/

To produce the RegionalSummary data in the output HDF5 file, the following
three options must all be provided in the call to ionstats alignment - run
with no arguments for usage info: --chip-origin, --chip-dim, --subregion-dim.
Additionally, the option --max-subregion-hp specifies the maximum HP length
that will be tracked in the regional summary


## 
## 4 - PerRead structure:
## 

The PerRead structure provides per-read data, and as such can be very large
It is implemented as a collection of HDF5 groups with the following datasets:

  n_read    Number of reads                             (uint32)
  n_flow    Number of flows                             (uint32)
  read_id   Vector or read ids                          (string)
  n_sub     Vector with number of subs per read         (uint16)
  ref_flow  Matrix of ref HP lengths per read per flow  (uint8)
  err_flow  Matrix of error sizes per read per flow     (uint8)

These groups are written out in sections per_read_per_flow/#/
