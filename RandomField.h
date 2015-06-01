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
			RandomField() : Function<dim>(dim+1) {};
			virtual ~RandomField() {};
			void generate( int type_of_distribution );									// generate instance, i.e. generate random vector
			void set_random_vector(std::vector<double>	&set_vector);
			static int get_stoch_dim();

		protected:
			std::vector<double>		random_vector;
			static int				stoch_dim;
			virtual double phi( double x, int n ) const = 0;	// n-th eigenfunction value of the K-L series
			virtual double ksi( int n ) const = 0;				// n-th eigenvalue of the K-L series
	};

	template<int dim> int 				RandomField<dim>::stoch_dim;


/*---------------------------------------------------------------------------*/
/*                           Class implementation                            */
/*---------------------------------------------------------------------------*/


	template< int dim >
	void RandomField<dim>::generate( int type_of_distribution )
	{
		RandomGenerator random_generator( type_of_distribution );

		random_generator.generate( random_vector );
	}


	template< int dim >
	void RandomField<dim>::set_random_vector( std::vector<double> &set_vector )
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
