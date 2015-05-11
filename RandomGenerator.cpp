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
	std::lognormal_distribution<double> lognormal_distribution(0.0,1.0);


	auto generate_uniform 	= std::bind ( uniform_distribution,   uniform_generator );
	auto generate_normal 	= std::bind ( normal_distribution,    uniform_generator );
	auto generate_lognormal	= std::bind ( lognormal_distribution, uniform_generator );


/*------------------------------ Class implementation -----------------------*/

	// constructor
	RandomGenerator::RandomGenerator( int type_of_distribution )
	{
		vector_size = 1;
		random_vector = new double[1];

		switch (type_of_distribution)
		{
			case Uniform:
				random_vector[0] = generate_uniform();
				break;
			case Normal:
				random_vector[0] = generate_normal();
				break;
			case LogNormal:
				random_vector[0] = generate_lognormal();
				break;
		}
	}


	// constructor
	RandomGenerator::RandomGenerator( int type_of_distribution, int n)
	{
		vector_size = n;
		random_vector = new double[n];

		switch (type_of_distribution)
		{
			case Uniform:
				for ( int i = 0; i < n; i++)
					random_vector[i] = generate_uniform();
				break;
			case Normal:
				for ( int i = 0; i < n; i++)
					random_vector[i] = generate_normal();
				break;
			case LogNormal:
				for ( int i = 0; i < n; i++)
					random_vector[i] = generate_lognormal();
				break;
		}
	}


	// return single random value
	double RandomGenerator::value()
	{
		return random_vector[0];
	}


	// return pointer to the random array
	double *RandomGenerator::vector( )
	{
		return random_vector;
	}

}//MonteCarlo

