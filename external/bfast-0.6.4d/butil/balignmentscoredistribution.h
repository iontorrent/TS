#ifndef BALIGNMENTSCOREDISTRIBUTION_H_
#define BALIGNMENTSCOREDISTRIBUTION_H_

typedef struct {
	int32_t num_cal;
	int32_t *hist;
	double from;
	double by;
	double to;
	int32_t length;
} CAL;

typedef struct {
	int32_t max_cals;
	CAL *cals;
} Dist;

void DistInitialize(Dist*);
int32_t DistAdd(Dist*, AlignedEnd*, double, double, double);
void DistPrint(Dist*, FILE*);
void DistFree(Dist*);
void CALInitialize(CAL*, int32_t, double, double, double);
int32_t CALAdd(CAL*, AlignedEnd*);
void CALPrint(CAL*, FILE*);
void CALFree(CAL*);

#endif
