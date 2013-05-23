/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "stats.h"
#define EC_CEILING 200

double poisson (double x, double Lam)
{
        if (Lam > EC_CEILING) {
            //fprintf(stderr, "%f %f\n", (float) x, (float) Lam);
            x *= EC_CEILING/Lam;
            Lam = EC_CEILING;
            //fprintf(stderr, "%f %f\n", (float) x, (float) Lam);
        }
        if (Lam < 1e-5) Lam = 1e-5;
        //printf("%f  ", (float) Lam);
        if (x < 0.0)
        {
                return 0.0;

        }
        if (x <= Lam) return 0;

        /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        /*
         * float iFac = 1.0;
         * float LamPow = 1.0;
         */
        //double                   ret = 0.0;
        double                  log_iFac = 0.0;
        double                  log_LamPow = 0.0;

        unsigned int    i, iMax = (unsigned int) x;
        /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        log_LamPow = log(Lam)*(iMax+1);
        for (i = 0; i <= iMax; i++)
        {

                /*
                 * if ((LamPow / iFac) < UNDER_LIMIT) { break;
                 * }
                 */
                //if ((log_LamPow - log_iFac) <= -20)
                //{
                  //      break;
                //}

                /*
                 * ret += LamPow/iFac;
                 */
//                ret += exp(log_LamPow - log_iFac);

                /*
                 * iFac *= (float) (i + 1);
                 * LamPow *= Lam;
                 */
                log_iFac += log((float) (i + 1));
                //log_LamPow += log(Lam);

        }

        //return ret * (float) exp(-Lam);
        return -(log_LamPow-log_iFac-Lam)/10.0;
}

void calc_score_hyp(int num_reads, int num_hyps, float **prob_matrix, float *score, int *count, float min_dif, float min_best)
{
	int num_sup[num_hyps];
	float total_error = 0.0;
	memset(num_sup, 0, sizeof(int)*num_hyps);
	int i, j, total_adj = 0;
	for (i = 0; i < num_reads; i++) {
		int bj = 0;
		float bs = 0, second_bs = 0;
		float *prob = prob_matrix[i];
		for (j = 0; j < num_hyps; j++) {
			if (prob[j] > bs) {
				second_bs = bs;
				bs = prob[j];
				bj = j;
			} else if (prob[j] > second_bs) {
				second_bs = prob[j];
			}
		}
		if (bs < min_best || bs-second_bs < min_dif) continue;
		total_adj++;
		total_error += (1.0-bs);
		num_sup[bj]++;
	}
	// total error is the expected number of observation 
	for (j = 0; j < num_hyps; j++) {
		score[j] = poisson(num_sup[j], total_error);	
	}
	if (count) {
		 for (j = 0; j < num_hyps; j++) count[j] = num_sup[j];
	}
}
