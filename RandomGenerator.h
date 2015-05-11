/*
 * RandomGenerator.h
 *
 *  Created on: Dec 16, 2014
 *      Author: viktor
 */

#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_

namespace MonteCarlo
{

	// types of supported distributions
	enum distributions { Uniform, Normal, LogNormal };

	class RandomGenerator
	{
		public:
			RandomGenerator( int type_of_distribution );
			RandomGenerator( int type_of_distribution, int n );
			virtual ~RandomGenerator() {};
			double value();
			double *vector( );
			int vector_size;

		private:
			double *random_vector;
	};


}//MonteCarlo


#endif /* RANDOMGENERATOR_H_ */
