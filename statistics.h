/*
 * statistics.h
 *
 *  Created on: Feb 15, 2015
 *      Author: viktor
 */

#ifndef STATISTICS_H_
#define STATISTICS_H_


// update statistics of scalar data
void update_statistics_s(	int old_n, 			int new_n,
							double &mean,		double &addit_mean,
							double &variance,	double &addit_variance )
{
	if ( new_n == 1 )
	{
		if ( old_n == 0 )
		{
			variance = 0.0;
			mean = addit_mean;
		}
		else
		{
			double b1 = 1.0 / ( old_n + 1 );
			variance = (old_n-1) * variance / ( old_n ) + b1 * ( mean - addit_mean ) * ( mean - addit_mean );
			mean = ( old_n * mean + addit_mean ) / ( old_n + 1 );
		}
	}
	else
	{
		if ( old_n == 0 )
		{
			variance = 0.0;
			mean = addit_mean;
		}
		else
		{
			double b1 = (double)( new_n * old_n ) / ( new_n + old_n - 1 ) / ( new_n + old_n );
			variance = ( (old_n-1) * variance + (new_n-1) * addit_variance ) / ( new_n + old_n - 1) + b1 * ( mean - addit_mean ) * ( mean - addit_mean );
			mean = ( old_n * mean + new_n * addit_mean ) / ( new_n + old_n );
		}
	}
}


// update statistics of vector data
void update_statistics_v(	int old_n, 						int new_n,
							BlockVector<double> &mean,		BlockVector<double> &addit_mean,
							BlockVector<double> &variance,	BlockVector<double> &addit_variance )
{
	if ( new_n == 1 )
	{
		if ( old_n == 0 )
		{
			variance = 0.0;
			mean = addit_mean;
		}
		else
		{
			for ( unsigned int i = 0; i < variance.size(); i++ )
			{
				variance[i] = (old_n-1)*variance[i]/old_n + ( mean[i] - addit_mean[i] )*( mean[i] - addit_mean[i] ) / ( old_n + 1 );
				mean[i] = ( old_n * mean[i] + addit_mean[i] ) / ( old_n + 1 );
			}
		}
	}
	else // new_n > 1
	{
		if ( old_n == 0 )
		{
			variance = addit_variance;
			mean = addit_mean;
		}
		else
		{
			double b1 = (double)( new_n * old_n ) / ( new_n + old_n - 1 ) / ( new_n + old_n );
			for ( unsigned int i = 0; i < variance.size(); i++ )
			{
				variance[i] = ( (old_n-1) * variance[i] + std::max(0,new_n-1) * addit_variance[i] ) / ( new_n + old_n - 1)
		              	  	  + b1 * ( mean[i] - addit_mean[i] ) * ( mean[i] - addit_mean[i] );
				mean[i] = ( old_n * mean[i] + new_n * addit_mean[i] ) / ( new_n + old_n );
			}

		}
	}
}



#endif /* STATISTICS_H_ */
