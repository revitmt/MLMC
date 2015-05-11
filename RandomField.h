/*
 * RandomField.h
 *
 *  Created on: Dec 16, 2014
 *      Author: viktor
 */

#ifndef RANDOMFIELD_H_
#define RANDOMFIELD_H_

#include <deal.II/base/function.h>	// function base class
#include "RandomGenerator.h"		// random generator

using namespace dealii;

namespace MonteCarlo
{

/*---------------------------------------------------------------------------*/
/*                           Class interface                                 */
/*---------------------------------------------------------------------------*/

	template< int dim >
	class RandomField : public Function<dim>
	{
		public:
			RandomField() {};
			virtual ~RandomField() {};
			void generate();									// generate instance, i.e. generate random vector
			void set_random_vector(std::vector<double>	&set_vector);
			static int get_stoch_dim();

		protected:
			std::vector<double>		random_vector;
			static distributions	type_of_distribution;
			static int				stoch_dim;
			virtual double phi( double x, int n ) const = 0;	// n-th eigenfunction value of the K-L series
			virtual double ksi( int n ) const = 0;				// n-th eigenvalue of the K-L series
	};

	template<int dim> distributions 	RandomField<dim>::type_of_distribution;
	template<int dim> int 				RandomField<dim>::stoch_dim;


/*---------------------------------------------------------------------------*/
/*                           Class implementation                            */
/*---------------------------------------------------------------------------*/


	template< int dim >
	void RandomField<dim>::generate()
	{
		RandomGenerator generate1( type_of_distribution, stoch_dim );
		double *vector = generate1.vector();

		random_vector.assign( vector, vector + stoch_dim );

		delete vector;
	}


	template< int dim >
	void RandomField<dim>::set_random_vector(std::vector<double>	&set_vector)
	{
		random_vector = set_vector;
	}


	template< int dim >
	int RandomField<dim>::get_stoch_dim()
	{
		return stoch_dim;
	}


}//MonteCarlo

#endif /* RANDOMFIELD_H_ */
