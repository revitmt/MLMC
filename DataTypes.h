/*
 * data_types.h
 *
 *  Created on: Feb 14, 2015
 *      Author: viktor
 */

#ifndef DATA_TYPES_H_
#define DATA_TYPES_H_

#include "DiscretizationData.h"

// TODO: obsolete
#include <deal.II/lac/vector.h>								// numerical vector of data --> Vector<double>
#include <deal.II/lac/sparse_matrix.h>						// sparse matrix denoted by a SparsityPattern --> SparseMatrix<double>

#include <deal.II/lac/block_vector.h>						// block vectors based on deal.II vectors --> BlockVector<double>
#include <deal.II/lac/block_sparse_matrix.h>				// block sparse matrix denoted by a BlockSparsityPattern --> BlockSparseMatrix<double>

#include <deal.II/lac/solver_cg.h>							// used in inverse matrix classes
#include <deal.II/lac/sparse_direct.h>						// SparseDirectUMFPACK

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
			solution_type( int l )
			{
				reinit(l);
			}
			virtual ~solution_type() {};
			void reinit( int l )
			{
				level = l;

			    const unsigned int n_u = DiscretizationData<dim>::dofs_per_block[level][0],
			                       n_p = DiscretizationData<dim>::dofs_per_block[level][1];

			    vector.reinit(2);
			    vector.block(0).reinit(n_u);
			    vector.block(1).reinit(n_p);
			    vector.collect_sizes();

			    vector = 0.0;
			};
			void interpolate_from( solution_type &input_vector )
			{
			    const unsigned int n_u = DiscretizationData<dim>::dofs_per_block[level][0],
			                       n_p = DiscretizationData<dim>::dofs_per_block[level][1];

				Vector<double> vec_1(n_u+n_p), vec_2(n_u+n_p);

				vec_1 = input_vector.vector;

				if ( DiscretizationData<dim>::intergrid_maps_ptr.size() != 0 )
					VectorTools::interpolate_to_different_mesh(	*(DiscretizationData<dim>::intergrid_maps_ptr[input_vector.level][level]),	vec_1,
																*(DiscretizationData<dim>::hanging_nodes_constraints_ptr[level]),			vec_2);
				else
					vector = 0.0;

				vector = vec_2;
			};
			void add( solution_type &input_vector, double multiplier = 1.0 )
			{
				solution_type<dim> tmp_vector(level);

				if ( level != input_vector.level )
					tmp_vector.interpolate_from(input_vector);
				else
					tmp_vector.vector = input_vector.vector;

				tmp_vector.vector *= multiplier;

				vector += tmp_vector.vector;
			};
			void subtract( solution_type &input_vector )
			{
				solution_type<dim> tmp_vector(level);
				tmp_vector.interpolate_from(input_vector);

				vector -= tmp_vector.vector;
			};
			double L1_norm( int component )
			{
				Vector<double> local_errors ( DiscretizationData<dim>::dof_handlers_ptr[level]->get_tria().n_active_cells() );

				ComponentSelectFunction<dim> weight(component, dim+1);

				VectorTools::integrate_difference (	*(DiscretizationData<dim>::dof_handlers_ptr[level]),
													vector,
													ZeroFunction<dim>(),
													local_errors,
													*(DiscretizationData<dim>::quadrature_formula),
													VectorTools::NormType::L1_norm,
													&weight );

				return local_errors.l1_norm();
			};
			double L2_norm( int component )
			{
				Vector<double> local_errors ( DiscretizationData<dim>::dof_handlers_ptr[level]->get_tria().n_active_cells() );

				ComponentSelectFunction<dim> weight(component, dim+1);

				VectorTools::integrate_difference (	*(DiscretizationData<dim>::dof_handlers_ptr[level]),
													vector,
													ZeroFunction<dim>(),
													local_errors,
													*(DiscretizationData<dim>::quadrature_formula),
													VectorTools::NormType::L2_norm,
													&weight );

				return local_errors.l2_norm();
			};
			double Linfty_norm( int component )
			{
				Vector<double> local_errors ( DiscretizationData<dim>::dof_handlers_ptr[level]->get_tria().n_active_cells() );

				ComponentSelectFunction<dim> weight(component, dim+1);

				VectorTools::integrate_difference (	*(DiscretizationData<dim>::dof_handlers_ptr[level]),
													vector,
													ZeroFunction<dim>(),
													local_errors,
													*(DiscretizationData<dim>::quadrature_formula),
													VectorTools::NormType::Linfty_norm,
													&weight );

				return local_errors.linfty_norm();
			};
			float estimate_posterior_error( int component )

			{
				Vector<float> estimated_error_per_cell ( DiscretizationData<dim>::triangulations_ptr[level]->n_active_cells() );

				FEValuesExtractors::Vector velocities(0);
				FEValuesExtractors::Scalar pressure(dim);
				ComponentMask mask = ( component < dim ) ? DiscretizationData<dim>::fe->component_mask(velocities) : DiscretizationData<dim>::fe->component_mask(pressure);

				KellyErrorEstimator<dim>::estimate(	*(DiscretizationData<dim>::dof_handlers_ptr[level]),
													QGauss<dim-1>(2),
													typename FunctionMap<dim>::type(),
													vector,
													estimated_error_per_cell,
													mask );

				return estimated_error_per_cell.l2_norm();
			}
			int sample;

		public:
			int level;
			BlockVector<double> vector;
	};




	/*---------------------------------------------------------------------------*/
	/*                    Auxiliary matrix type class                            */
	/*---------------------------------------------------------------------------*/
	template< int dim >
	class matrix_type
	{
		public:
			matrix_type() {};
			matrix_type( int l ) { reinit(l); }
			virtual ~matrix_type() {};
			void reinit( int l )
			{
				level = l;
				matrix.reinit (*(DiscretizationData<dim>::sparsity_patterns_ptr[level]));
			};

		public:
			int level;
			BlockSparseMatrix<double> matrix;
	};






	/*----------------------------------------------------------------------------*/
	/* Inverse matrix class used to efficiently multiply inverse matrix by vector */
	/*----------------------------------------------------------------------------*/
	template <class Matrix, class Preconditioner>
	class InverseMatrix : public Subscriptor
	{
		public:
			InverseMatrix( const Matrix &m, const Preconditioner &preconditioner ) : matrix(&m), preconditioner (&preconditioner) {};
			void vmult( Vector<double> &dst, const Vector<double> &src) const
			{
				SolverControl solver_control( src.size(), 1e-6*src.l2_norm() );
				SolverCG<>    cg(solver_control);

				dst = 0;

				cg.solve ( *matrix, dst, src, *preconditioner );
			};

		private:
			const SmartPointer<const Matrix> matrix;
			const SmartPointer<const Preconditioner> preconditioner;
	};








/*---------------------------------------------------------------------------*/
/*                        block Schur preconditioner                         */
/*---------------------------------------------------------------------------*/
	template <class PreconditionerA, class PreconditionerMp>
	class BlockSchurPreconditioner : public Subscriptor
	{
		public:
			BlockSchurPreconditioner(	const BlockSparseMatrix<double>								&S,
										const InverseMatrix<SparseMatrix<double>,PreconditionerMp>	&Mpinv,
										const PreconditionerA										&Apreconditioner )
            						:
            						system_matrix		(&S),
            						m_inverse			(&Mpinv),
            						a_preconditioner	(Apreconditioner),
            						tmp					(S.block(1,1).m()) {};
			void vmult(	BlockVector<double>			&dst,
						const BlockVector<double>	&src) const
			{
				// Form u_new = A^{-1} * u
				a_preconditioner.vmult( dst.block(0), src.block(0) );
			    // Form tmp = -B * u_new + p (SparseMatrix::residual does precisely this)
			    system_matrix->block(1,0).residual( tmp, dst.block(0), src.block(1) );
			    // Change sign in tmp
			    tmp *= -1;
			    // Multiply by approximate Schur complement, i.e. a pressure mass matrix
			    m_inverse->vmult ( dst.block(1), tmp );
			}
		private:
			const SmartPointer<const BlockSparseMatrix<double>>								system_matrix;
			const SmartPointer<const InverseMatrix<SparseMatrix<double>,PreconditionerMp>>	m_inverse;
			const PreconditionerA															&a_preconditioner;
			mutable Vector<double>															tmp;
	  };


#endif /* DATA_TYPES_H_ */
