/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifdef DEBUG_TRACE
    #include <iostream>
    #include <iomanip>
#endif

// DEBUG_TRACE_MATRIX turns on printing the trace matrix when backtrace () function is called
// #define DEBUG_TRACE_MATRIX 1


namespace genstr
{

/////////////////////////////////////////////////////////
// Score

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
void Score <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::dynamic ()
{
    maxPos1_ = maxPos2_ = UINT_MAX;
    bestWeight_ = (ValueType) 0;

    init_dynamic ();
    // for every position in seq1
    for (unsigned pos1 = 0; pos1 < len1_; pos1 ++)
        // perform inner loop
        dynamic_seq2 (pos1);
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
void Score <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::dynamic_seq2 (unsigned pos1)
{
    // initialize what is needed for column calculation
    init_dynamic_col (pos1);
    for (unsigned pos2 = 0; pos2 < len2_; pos2 ++)
    {
        ValueType weight = dynamic_position (pos1, pos2);
        if (weight > bestWeight_)
            bestWeight_ = weight, maxPos1_ = pos1, maxPos2_ = pos2;
    }
#ifdef DEBUG_TRACE
    std::cout << std::endl;
#endif
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
ValueType Score <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::dynamic_position (unsigned pos1, unsigned pos2)
{
    // compute best scores for three paths that can lead to current cell (at pos1, pos2 of dynamic matrix)
    ValueType score = score_diag  (pos1, pos2);
    score = std::max (score, score_1 (pos1, pos2));
    score = std::max (score, score_2 (pos1, pos2));
    return score;
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
void Score <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::configure (WeightMatrix <DestSymbolType, ValueType, AveType>* weight_matrix, GapCost<GapType>* gap_cost1, GapCost<GapType>* gap_cost2, Translator<Src1SymbolType, DestSymbolType>* tr1, Translator<Src2SymbolType, DestSymbolType>* tr2)
{
    weight_matrix_ = weight_matrix;
    gap_cost_eval_1_ = gap_cost1;
    gap_cost_eval_2_ = gap_cost2;
    if (tr1 != NULL) translator1_ = tr1;
    else translator1_ = &def_translator1_;
    if (tr2 != NULL) translator2_ = tr2;
    else translator2_ = &def_translator2_;
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
ValueType Score <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::eval (const Src1SymbolType* seq1, unsigned len1, const Src2SymbolType* seq2, unsigned len2)
{
    seq1_ = seq1;
    seq2_ = seq2;
    len1_ = len1;
    len2_ = len2;
    dynamic ();
    return bestWeight_;
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
ValueType Align <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::dynamic_position (unsigned pos1, unsigned pos2)
{
    // compute best scores for three paths that can lead to current cell (at pos1, pos2 of dynamic matrix)
    ValueType sd = this->score_diag (pos1, pos2);
    ValueType s1 = this->score_1 (pos1, pos2);
    ValueType s2 = this->score_2 (pos1, pos2);
    ValueType best = (ValueType) 0;
    PATH_DIR dir = TRACE_STOP;
    if (sd > 0 && sd > s1 && sd > s2)
    {
        best = sd;
        dir = ALONG_DIAG;
#ifdef DEBUG_TRACE
        std::cout << "\\";
#endif
    }
    else if (s1 > 0 && s1 > s2)
    {
        best = s1;
        dir = ALONG_FIRST;
#ifdef DEBUG_TRACE
        std::cout << "|";
#endif
    }
    else if (s2 > 0)
    {
        best = s2;
        dir = ALONG_SECOND;
#ifdef DEBUG_TRACE
        std::cout << "-";
#endif
    }
#ifdef DEBUG_TRACE
    else
        std::cout << " ";
    std::cout << std::setw (3) << std::left << best << std::setw (0) << std::flush;
#endif
    // update trace matrix
    trace_.put (pos1, pos2, dir);
    return best;
}

template <class Src1SymbolType, class Src2SymbolType, class DestSymbolType, class ValueType, class AveType, class GapType>
void Align <Src1SymbolType, Src2SymbolType, DestSymbolType, ValueType, AveType, GapType>
::backtrace ()
{
#ifdef DEBUG_TRACE_MATRIX
    trace_.print (this->maxPos1_, this->maxPos2_);
#endif

    if (this->maxPos1_ == UINT_MAX) return;
    if (!result_)
    {
        this->result_ = new Alignment;
        unsigned pos1 = this->maxPos1_;
        unsigned pos2 = this->maxPos2_;
        unsigned batch_end = UINT_MAX;
        int prev_direction ;
        int direction = TRACE_STOP;

        while (pos1 != UINT_MAX && pos2 != UINT_MAX) // this relays on the assumption that --((unsigned) 0) == UINT_MAX
        {
            prev_direction = direction;
            direction  = this->trace_.get (pos1, pos2);
            switch (direction)
            {
                case ALONG_DIAG:
                    if (prev_direction != ALONG_DIAG) batch_end = pos1;
                    pos1 = (pos1 < this->translator1_->unit ()) ? UINT_MAX: pos1 - this->translator1_->unit ();
                    pos2 = (pos2 < this->translator2_->unit ()) ? UINT_MAX: pos2 - this->translator2_->unit ();
                    break;
                case ALONG_FIRST:
                    if (batch_end != UINT_MAX && prev_direction == ALONG_DIAG)
                        this->result_->push_front (Batch (pos1 + 1, pos2 + 1, batch_end - pos1));
                    pos1 --;
                    break;
                case ALONG_SECOND:
                    if (batch_end != UINT_MAX && prev_direction == ALONG_DIAG)
                        this->result_->push_front (Batch (pos1 + 1, pos2 + 1, batch_end - pos1));
                    pos2 --;
                    break;
                default:
                    if (batch_end != UINT_MAX && prev_direction == ALONG_DIAG)
                        this->result_->push_front (Batch (pos1 + 1, pos2 + 1, batch_end - pos1));
                    pos1 = UINT_MAX; // terminate
            }
        }
        if (batch_end != UINT_MAX && direction == ALONG_DIAG)
            this->result_->push_front (Batch (pos1 + 1, pos2 + 1, batch_end - pos1)); // this relays on assumption that (UINT_MAX + 1) == 0
    }
}


} // namespace genstr

