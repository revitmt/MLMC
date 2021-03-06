/*
 * InputData.h
 *
 *  Created on: Dec 16, 2014
 *      Author: viktor
 */

#ifndef INPUTDATA_H_
#define INPUTDATA_H_

#include "RandomField.h"
#include "SharedData.h"

using namespace MonteCarlo;


namespace deterministic_solver
{
	#define PI 3.1415926535897

/*---------------------------------------------------------------------------*/
/*                           Class interface                                 */
/*---------------------------------------------------------------------------*/


/*-------------------------- Right hand side function -----------------------*/

	template <int dim>
	class RightHandSide : public RandomField<dim>
	{
		public:
			RightHandSide () : RandomField<dim>() {};
			virtual ~RightHandSide () {};
			virtual double value (	const Point<dim>	&p,
									const unsigned int	component = 0) const;
			virtual void value_list (	const std::vector<Point<dim> >	&points,
										std::vector<double>				&values,
										const unsigned int				component = 0)  const;

		private:
			virtual double phi( double x, int n ) const;	// n-th eigenfunction value of the K-L series
			virtual double ksi( int n ) const;				// n-th eigenvalue of the K-L series
	};

/*------------------------------ Boundary values ----------------------------*/

	template <int dim>
	class BoundaryValues : public RandomField<dim>
	{
		public:
			BoundaryValues () : RandomField<dim>() {};
			virtual ~BoundaryValues () {};
			virtual double value (	const Point<dim>	&p,
									const unsigned int	component = 0) const;

		private:
			virtual double phi( double x, int n ) const;	// n-th eigenfunction value of the K-L series
			virtual double ksi( int n ) const;				// n-th eigenvalue of the K-L series
	};

/*---------------------------- Coefficient function -------------------------*/

	template <int dim>
	class Coefficient : public RandomField<dim>
	{
		public:
			Coefficient ();
			virtual ~Coefficient () {};
			virtual double value (	const Point<dim>	&p,
	                        		const unsigned int	component = 0)  const;
			virtual void value_list (	const std::vector<Point<dim> >	&points,
										std::vector<double>				&values,
										const unsigned int				component = 0)  const;
			std::vector<double>	get_random_vector() const;
			std::vector<double>	get_coefficients() const;
			static double		get_cor_length();
			void 				initialize( double cor_length );						// set type of distribution, L_c and stoch_dim

		private:
			static double	Lc;						// correlation length
			static double	Lp;
			static double	L;
			static vector<double>	eigenvalue;
			virtual double	phi( double x, int n ) const;	// n-th eigenfunction value of the K-L series
			virtual double	ksi( int n ) const;				// n-th eigenvalue of the K-L series
	};

	template<int dim>	double			Coefficient<dim> :: Lc;
	template<int dim>	double			Coefficient<dim> :: Lp;
	template<int dim>	double			Coefficient<dim> :: L;
	template<int dim> 	vector<double>	Coefficient<dim> :: eigenvalue;



/*---------------------------------------------------------------------------*/
/*                           Class implementation                            */
/*---------------------------------------------------------------------------*/
#define random_vector			this->random_vector
#define generate_random_vector	this->generate_random_vector
#define stoch_dim				RandomField<dim>::stoch_dim
#define type_of_distribution	RandomField<dim>::type_of_distribution

#define set_random_vector		this->set_random_vector


/*-------------------------- Right hand side function -----------------------*/

	template <int dim>
	double RightHandSide<dim>::value (	const Point<dim> &p,
										const unsigned int component) const
	{
		double x = p[0];
		double y = p[1];
		return cos(x) * sin(y);
	}


	template <int dim>
	void RightHandSide<dim>::value_list (	const std::vector<Point<dim> >	&points,
											std::vector<double>				&values,
											const unsigned int				component) const
	{
		Assert (values.size() == points.size(),	ExcDimensionMismatch (values.size(), points.size()));
		Assert (component == 0,	ExcIndexRange (component, 0, 1));

		const unsigned int n_points = points.size();

		for ( unsigned int i = 0; i < n_points; ++i)
			values[i] = value( points[i] );
	}


	template< int dim >
	double RightHandSide<dim>::phi( double x, int n ) const
	{
		return 0.0;
	}


	template< int dim >
	double RightHandSide<dim>::ksi( int n ) const
	{
		return 0.0;
	}




/*------------------------------ Boundary values ----------------------------*/

	template <int dim>
	double BoundaryValues<dim>::value (	const Point<dim> &p,
										const unsigned int component) const
	{
		return 0.0;
	}


	template< int dim >
	double BoundaryValues<dim>::phi( double x, int n ) const
	{
		return 0.0;
	}


	template< int dim >
	double BoundaryValues<dim>::ksi( int n ) const
	{
		return 0.0;
	}



/*---------------------------- Coefficient function -------------------------*/

	template <int dim>
	Coefficient<dim>::Coefficient() : RandomField<dim>()
	{}


	template <int dim>
	void Coefficient<dim>::initialize( double cor_length )
	{
		type_of_distribution = Uniform;
		Lc = cor_length;
		Lp = std::max(1.0,2*Lc);
		L  = Lc / Lp;

//		for ( stoch_dim = 1; ksi(stoch_dim) > 0.01; stoch_dim++ ){};

		stoch_dim = 50;

		eigenvalue.resize(stoch_dim);
		for ( int i = 0; i < stoch_dim-1; i++ )
			eigenvalue[i] = ksi( i );
	}


	template< int dim >
	double Coefficient<dim>::phi( double x, int n ) const
	{
		if ( n == 1 )
			return sqrt(0.5);

		if ( n & 1 )	// odd
			return cos( PI * x * floor( n / 2.0 ) / Lp );
		else			// even
			return sin( PI * x * floor( n / 2.0 ) / Lp );
	}


	template< int dim >
	double Coefficient<dim>::ksi( int n ) const
	{
		if ( n == 1 )
			return sqrt( sqrt(PI) * L );
		else
		{
			double tmp = PI * L * floor( n / 2.0 );
			return sqrt( sqrt(PI) * L ) * exp( - tmp * tmp / 8.0 );
		}

	}


	template <int dim>
	double Coefficient<dim>::value (	const Point<dim> &p,
										const unsigned int component) const
	{
		double value;
		double x = p[0];
//		double y = p[1];

		value = 1.0;
		for ( int j = 0; j < stoch_dim; j++ )
			value += random_vector[j] * ksi(j+1) * phi(x, j+1);
		value = 0.5 + exp(value);

		return value;
	}


	template <int dim>
	void Coefficient<dim>::value_list (	const std::vector<Point<dim> >	&points,
										std::vector<double>				&values,
										const unsigned int				component) const
	{
		Assert (values.size() == points.size(),	ExcDimensionMismatch (values.size(), points.size()));
		Assert (component == 0,	ExcIndexRange (component, 0, 1));

		const unsigned int n_points = points.size();

		for ( unsigned int i = 0; i < n_points; ++i)
			values[i] = value( points[i] );
	}


	template <int dim>
	std::vector<double> Coefficient<dim>::get_random_vector() const
	{
		return random_vector;
	}


	template <int dim>
	std::vector<double> Coefficient<dim>::get_coefficients() const
	{
		std::vector<double>  coefficients;

		coefficients.resize(stoch_dim);
		for ( int j = 0; j < stoch_dim; j++ )
			coefficients[j] = random_vector[j] * ksi(j+1);

		return coefficients;
	}


	template <int dim>
	double Coefficient<dim>::get_cor_length()
	{
		return Lc;
	}


#undef random_vector
#undef generate_random_vector
#undef stoch_dim
#undef type_of_distribution

#undef set_random_vector












}//deterministic_solver

#endif /* INPUTDATA_H_ */
