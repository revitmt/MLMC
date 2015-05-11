/*
 * data_types.h
 *
 *  Created on: Feb 14, 2015
 *      Author: viktor
 */

#ifndef DATA_TYPES_H_
#define DATA_TYPES_H_

#include "DiscretizationData.h"

#include <deal.II/lac/vector.h>								// numerical vector of data --> Vector<double>
#include <deal.II/lac/sparse_matrix.h>						// sparse matrix denoted by a SparsityPattern --> SparseMatrix<double>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/error_estimator.h>				// Kelly error estimator --> solution type


using namespace MonteCarlo;

/*---------------------------------------------------------------------------*/
/*                    Auxiliary solution type class                          */
/*---------------------------------------------------------------------------*/
	template< int dim >
	class solution_type
	{
		public:
			solution_type() {};
			solution_type( int l ) :	level(l),
										vector(DiscretizationData<dim>::dof_handlers_ptr[level]->n_dofs()) {};
			virtual ~solution_type() {};
			void reinit( int l )
			{
				level = l;
				vector.reinit( DiscretizationData<dim>::dof_handlers_ptr[level]->n_dofs() );
				vector = 0.0;
			};
			void interpolate_from( solution_type &input_vector )
			{
				if ( DiscretizationData<dim>::intergrid_maps_ptr.size() != 0 )
					VectorTools::interpolate_to_different_mesh(	*(DiscretizationData<dim>::intergrid_maps_ptr[input_vector.level][level]),	input_vector.vector,
																*(DiscretizationData<dim>::hanging_nodes_constraints_ptr[level]),			vector);
				else
					vector = 0.0;
			};
			void add( solution_type &input_vector, double multiplier = 1.0 )
			{
				solution_type<dim> tmp_vector(level);
				tmp_vector.interpolate_from(input_vector);
				tmp_vector.vector *= multiplier;

				vector += tmp_vector.vector;
			};
			void subtract( solution_type &input_vector )
			{
				solution_type<dim> tmp_vector(level);
				tmp_vector.interpolate_from(input_vector);

				vector -= tmp_vector.vector;
			};
			double L1_norm()
			{
				Vector<double> local_errors ( DiscretizationData<dim>::dof_handlers_ptr[level]->get_tria().n_active_cells() );

				VectorTools::integrate_difference (	*(DiscretizationData<dim>::dof_handlers_ptr[level]),
													vector,
													ZeroFunction<dim>(),
													local_errors,
													*(DiscretizationData<dim>::quadrature_formula),
													VectorTools::NormType::L1_norm);

				return local_errors.l1_norm();
			};
			double L2_norm()
			{
				Vector<double> local_errors ( DiscretizationData<dim>::dof_handlers_ptr[level]->get_tria().n_active_cells() );

				VectorTools::integrate_difference (	*(DiscretizationData<dim>::dof_handlers_ptr[level]),
													vector,
													ZeroFunction<dim>(),
													local_errors,
													*(DiscretizationData<dim>::quadrature_formula),
													VectorTools::NormType::L2_norm);

				return local_errors.l2_norm();
			};
			double Linfty_norm()
			{
				Vector<double> local_errors ( DiscretizationData<dim>::dof_handlers_ptr[level]->get_tria().n_active_cells() );

				VectorTools::integrate_difference (	*(DiscretizationData<dim>::dof_handlers_ptr[level]),
													vector,
													ZeroFunction<dim>(),
													local_errors,
													*(DiscretizationData<dim>::quadrature_formula),
													VectorTools::NormType::Linfty_norm);

				return local_errors.linfty_norm();
			};
			float estimate_posterior_error()

			{
				Vector<float> estimated_error_per_cell ( DiscretizationData<dim>::triangulations_ptr[level]->n_active_cells() );
				KellyErrorEstimator<dim>::estimate(	*(DiscretizationData<dim>::dof_handlers_ptr[level]),
													QGauss<dim-1>(2),
													typename FunctionMap<dim>::type(),
													vector,
													estimated_error_per_cell );

				return estimated_error_per_cell.l2_norm();
			}
			int sample;

		public:
			int level;
			Vector<double> vector;
	};




	/*---------------------------------------------------------------------------*/
	/*                    Auxiliary matrix type class                            */
	/*---------------------------------------------------------------------------*/
		template< int dim >
		class matrix_type
		{
			public:
				matrix_type() {};
				matrix_type( int l ) :	level(l),
										matrix( *(DiscretizationData<dim>::sparsity_patterns_ptr[level]) ) {};
				virtual ~matrix_type() {};
				void reinit( int l )
				{
					level = l;
					matrix.reinit (*(DiscretizationData<dim>::sparsity_patterns_ptr[level]));
				};

			public:
				int level;
				SparseMatrix<double> matrix;
		};



#endif /* DATA_TYPES_H_ */
