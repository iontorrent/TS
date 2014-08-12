/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SIGNALAVERAGER_H
#define SIGNALAVERAGER_H


class SignalAverager {

public:
	SignalAverager(bool *mask,int numElements)
	{
		int num_avg = 0;
		
		// figure out how many are set in the mask
		for (int i=0;i < numElements;i++)
		{
			if (mask[i])
				num_avg++;
		}

		nAvg = num_avg;
		avg_indicies = new int[nAvg];

		num_avg = 0;
		for (int i=0;(i < numElements) && (num_avg < nAvg);i++)
		{
			if (mask[i])
				avg_indicies[num_avg++] = i;
		}
		
		next = NULL;
	}

	void SetRow(int _row)
	{
		row = _row;
	}

	void SetCol(int _col)
	{
		col = _col;
	}

	int GetRow(void)
	{
		return(row);
	}

	int GetCol(void)
	{
		return(col);
	}
	
	DATA_TYPE GetAverage(DATA_TYPE *cmatrix)
	{
		DATA_TYPE ret = 0.0f;
		
		for (int i=0;i < nAvg;i++)
			ret += cmatrix[avg_indicies[i]];
			
		return(ret/nAvg);
	}
	
	// restricts the average to elements that are represented in class_mask
	// can be used to get signal along a sidewall for example
	DATA_TYPE GetAverage(DATA_TYPE *cmatrix,unsigned char *boundary_class,BoundaryClass class_mask)
	{
		DATA_TYPE ret = 0.0f;
		int ntot=0;
		
		for (int i=0;i < nAvg;i++)
		{
			if (boundary_class[avg_indicies[i]] & class_mask)
			{
				ret += cmatrix[avg_indicies[i]];
				ntot++;
			}
		}
			
		return(ret/ntot);
	}
	
	
	void SetNext(SignalAverager *next_avg)
	{
		next = next_avg;
	}

	SignalAverager *GetNext(void)
	{
		return(next);
	}
	
	~SignalAverager()
	{
		delete avg_indicies;
	}

private:
	int nAvg;
	int *avg_indicies;
	int row,col;
	SignalAverager *next;

};

#endif // SIGNALAVERAGER_H

