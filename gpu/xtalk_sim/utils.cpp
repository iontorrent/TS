
#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "xtalk_sim.h"

Coordinate GetWellLocation(int row,int col,DATA_TYPE pitch)
{
	// find the center location (in microns) of a well in a hex-packed array
	// given a hex well spacing and the coordinates of the well
	// The hex pattern is assumed to be oriented such that columns contain
	// hexagonal wells stacked directly on-top of each other in a straight line,
	// but rows appear jagged (as they are in a proton device)
	//
	// it is assumed that row=0 and col=0 is located at xpos=0 and ypos=0
	// pitch is the distance between centers of adjacent hexagonal wells
	Coordinate ret;

	ret.y = pitch*row;
	if ((col % 2)!=0)
		ret.y = ret.y + pitch/2.0f;
		
	ret.x = sin(M_PI/3.0f)*pitch*col;
	ret.z = 0.0f;
	
	return(ret);
}

//    wall_frac = 0.82;
HexagonDescriptor GetHexagon(Coordinate loc,DATA_TYPE pitch,DATA_TYPE well_fraction)
{
	HexagonDescriptor descr;
	DATA_TYPE spd3 = sin(M_PI/3.0f);
	DATA_TYPE x_vert[] = {1.0f, 0.5f, -0.5f, -1.0f, -0.5f,  0.5f, 1.0f};
	DATA_TYPE y_vert[] = {0.0f, spd3,  spd3,  0.0f, -spd3, -spd3, 0.0f};
	
	for (int i=0;i < 7;i++)
	{
		descr.vert[i].y = y_vert[i]*(pitch/(2.0f*spd3))*well_fraction + loc.y;
		descr.vert[i].x = x_vert[i]*(pitch/(2.0f*spd3))*well_fraction + loc.x;
		descr.vert[i].z = 0.0f;
	}

	return(descr);
}

DATA_TYPE FindNextHexagonCrossing(HexagonDescriptor descr,DATA_TYPE xloc,DATA_TYPE prevy,DATA_TYPE maxy)
{
	DATA_TYPE ret = maxy;
	
	for (int i=0;i < 6;i++)
	{
		DATA_TYPE x1,y1,x2,y2;
		
		x1 = descr.vert[i].x;
		x2 = descr.vert[i+1].x;
		
		// test if this line actually crosses xloc
		if (((x1 < xloc) && (x2 > xloc)) 
		    || ((x1 > xloc) && (x2 < xloc))
			|| ((x1 == xloc) && (x2 != xloc))
			|| ((x1 != xloc) && (x2 == xloc)))
		{
			y1 = descr.vert[i].y;
			y2 = descr.vert[i+1].y;
			
			// it does cross...so calculate at which y value it crosses
			DATA_TYPE ycross = y1+(y2-y1)*(xloc - x1)/(x2-x1);
			
			if ((ycross < ret) && (ycross > prevy))
				ret = ycross;
		}
	}
	
	return(ret);
}

// tests if the described hexagon is fully within the 2-d x-y area of the simulation (where there are ni-by-nj elements
// and each element is dx-by-dy in size.  All points of the hexagon must be completely inside the 2-d area by more than
// the specified margin.  If the hexagon passes this test, all elements that fall within the hexagon are marked at true
// in the mask and the number of elements marked as true is returned.  Otherwise, 0 is returned and mask is not altered.
int MarkInHexagon(HexagonDescriptor descr,bool *mask,int ni,int nj,DATA_TYPE dx,DATA_TYPE dy,DATA_TYPE margin)
{
	int ret = 0;
	DATA_TYPE xmax,ymax;
	Coordinate max_test,min_test;
	
	xmax = (ni-1)*dx;
	ymax = (nj-1)*dy;

	// find the box that contains the hexagon entirely, which is useful for
	// limiting the number of elements we need to test
	max_test.x = -DATA_TYPE_MAX;
	max_test.y = -DATA_TYPE_MAX;
	min_test.x = DATA_TYPE_MAX;
	min_test.y = DATA_TYPE_MAX;
	
	for (int i=0;i < 7;i++)
	{
		if (descr.vert[i].x < margin)
			return(0);
		if (descr.vert[i].x > (xmax-margin))
			return(0);
		if (descr.vert[i].y < margin)
			return(0);
		if (descr.vert[i].y > (ymax-margin))
			return(0);
			
		if (descr.vert[i].x > max_test.x)
			max_test.x = descr.vert[i].x;
		if (descr.vert[i].y > max_test.y)
			max_test.y = descr.vert[i].y;
		if (descr.vert[i].x < min_test.x)
			min_test.x = descr.vert[i].x;
		if (descr.vert[i].y < min_test.y)
			min_test.y = descr.vert[i].y;
	}

	int starti,startj,endi,endj;
	
	starti = (int)(min_test.x / dx)-1;
	startj = (int)(min_test.y / dy)+1;
	endi = (int)(max_test.x / dx + 0.5f)-1;
	endj = (int)(max_test.y / dy + 0.5f)+1;
	
	// go through each column, and find those rows that are inside the hexagon
	for (int i=starti;i <=endi;i++)
	{
		// we are guaranteed to start outside the hexagon
		// find the first crossing of the perimeter
		bool inside = false;
		DATA_TYPE xloc = dx * i;
		DATA_TYPE next_yloc = FindNextHexagonCrossing(descr,xloc,min_test.y-0.5f,max_test.y+0.5f);
		DATA_TYPE prev_yloc = min_test.y;
		
		while(next_yloc < (max_test.y+0.5f))
		{
//			printf("%f,%f,%f\n",xloc,next_yloc,prev_yloc);
			
			if (!inside)
			{
				// we found a crossing that takes us inside...we need to find the next
				// crossing before we know what the other boundary of 'inside' is
				inside = true;
			}
			else
			{
				// we just found out where an inside-to-outside boundary is
				inside = false;

				// anything with a y coordinate between prev and next yloc is inside the
				// polygon.
				for (int j=startj;j < endj;j++)
				{
					DATA_TYPE yloc = dy * j;
					
					if ((yloc > prev_yloc) && (yloc < next_yloc))
					{
						ret++;
						mask[i+j*ni] = true;
					}
				}
			}
		
			prev_yloc = next_yloc;
			next_yloc = FindNextHexagonCrossing(descr,xloc,next_yloc,(max_test.y+0.5f));
		}
	}
	
	return(ret);
}





