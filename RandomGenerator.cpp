/*
 * RandomGenerator.h
 *
 *  Created on: Dec 16, 2014
 *      Author: viktor
 */

#include "RandomGenerator.h"

#include <chrono>
#include <random>
#include <functional>   // std::bind
#include <math.h>

namespace MonteCarlo
{


/*---------------------- Generators and distributions -----------------------*/

	// construct seed
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	// mersenne_twister_engine uniform generator
	std::mt19937 uniform_generator(seed);

	// uniform distribution with left and right endpoints
	std::uniform_real_distribution<double> uniform_distribution( -sqrt(3.0), sqrt(3.0) );

	// normal distribution with mean and std deviation
	std::normal_distribution<double> normal_distribution (0.0, 1.0);

	// lognormal distribution
	std::lognormal_distribution<double> lognormal_distribution(0.0, 1.0);


	auto generate_uniform 	= std::bind ( uniform_distribution,   uniform_generator );
	auto generate_normal 	= std::bind ( normal_distribution,    uniform_generator );
	auto generate_lognormal	= std::bind ( lognormal_distribution, uniform_generator );


/*------------------------------ Class implementation -----------------------*/

	void RandomGenerator::generate( double &value )
	{
		switch ( distribution )
		{
			case Uniform:
				value = generate_uniform();
				break;
			case Normal:
				value = generate_normal();
				break;
			case LogNormal:
				value = generate_lognormal();
				break;
		}
	}



	void RandomGenerator::generate( std::vector<double> &vector )
	{
		int n = vector.size();

		switch ( distribution )
		{
			case Uniform:
				for ( int i = 0; i < n; i++)
					vector[i] = generate_uniform();
				break;
			case Normal:
				for ( int i = 0; i < n; i++)
					vector[i] = generate_normal();
				break;
			case LogNormal:
				for ( int i = 0; i < n; i++)
					vector[i] = generate_lognormal();
				break;
		}
	}

}//MonteCarlo

