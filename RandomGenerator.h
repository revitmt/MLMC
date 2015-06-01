/*
 * RandomGenerator.h
 *
 *  Created on: Dec 16, 2014
 *      Author: viktor
 */

#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_

#include <vector>

namespace MonteCarlo
{

	// types of supported distributions
	enum distributions { Uniform, Normal, LogNormal };

	class RandomGenerator
	{
		public:
			RandomGenerator( int type_of_distribution ) : distribution(type_of_distribution) {};
			virtual ~RandomGenerator() {};
			void generate( double &value );
			void generate( std::vector<double> &vector );

		private:
			int distribution;
	};


}//MonteCarlo


#endif /* RANDOMGENERATOR_H_ */
