/*
 * Navier_Stokes.h
 *
 *  Created on: Dec 13, 2014
 *      Author: viktor
 */

#ifndef Navier_Stokes_H_
#define Navier_Stokes_H_

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_base.h>
//#include <deal.II/base/timer.h>					// CPU time


#include <deal.II/lac/full_matrix.h>			// classical rectangular matrix
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_relaxation.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>				// Incomplete LU (ILU) decomposition of a sparse matrix -> SparseILU<double> pmass_preconditioner

#include <deal.II/dofs/dof_accessor.h>			// access information related to dofs while iterating trough cells, faces, etc --> get_dof_indices ( local_dof_indices )

#include <deal.II/fe/fe_values.h>				// finite element functions evaluated in quadrature points of a cell --> FEValues<dim>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>

#include "DiscretizationData.h"
#include "DataTypes.h"
#include "InputData.h"
#include "Output.h"

#include "my_timer.h"

#include "Stokes.h"

#include <iostream>
#include <fstream>

using namespace dealii;




/*---------------------------------------------------------------------------*/
/*                           Class interface                                 */
/*---------------------------------------------------------------------------*/

	template <int dim>
	struct InnerPreconditioner;

	template<>
	struct InnerPreconditioner<2>
	{
		typedef SparseDirectUMFPACK type;
	};

	template<>
	struct InnerPreconditioner<3>
	{
		typedef SparseILU<double> type;
	};


	template <int dim>
	class Navier_Stokes
	{
		private:
			struct problem_info
			{
				int		newton_iterations;
				int		total_iterations;
				double	assemble_CPU_time;
				double	solve_CPU_time;
				double	total_CPU_time;
				double 	init_residual;

				problem_info(){reset();};

				void reset()
				{
					total_iterations = newton_iterations = 0;
					total_CPU_time = 0.0;
					assemble_CPU_time = 0.0;
					solve_CPU_time = 0.0;
				};

				void print( std::ostream &out )
				{
					out << std::scientific << std::setprecision(4);
					out << init_residual
						<< " " << total_iterations
						<< " " << newton_iterations
					    << " " << assemble_CPU_time
						<< " " << solve_CPU_time
						<< " " << total_CPU_time;
				}
			};
			struct ScratchData
			{
				std::vector<Vector<double>>	function_values;
				FEValues<dim>	 	fe_values;
				ScratchData () : function_values( n_q_points, Vector<double>(dim+1) ), fe_values ( *fe, *quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values  ) {};
				ScratchData (const ScratchData &scratch) : function_values(scratch.function_values),
															 fe_values(	scratch.fe_values.get_fe(),
																		scratch.fe_values.get_quadrature(),
																		scratch.fe_values.get_update_flags()) {}
			};
			struct PerTaskRhsData
			{
				Vector<double>            cell_rhs;
				std::vector<unsigned int> local_dof_indices;
				PerTaskRhsData ( const int &dof_per_cell ) : cell_rhs(dof_per_cell), local_dof_indices(dof_per_cell){};
			};
			struct PerTaskMatrixData
			{
				FullMatrix<double>        cell_matrix;
				std::vector<unsigned int> local_dof_indices;
				PerTaskMatrixData ( const int &dof_per_cell ) : cell_matrix(dof_per_cell,dof_per_cell), local_dof_indices(dof_per_cell){};
			};

		public:
			Navier_Stokes(){};
			Navier_Stokes( int lev, double lin_sol_error );
			virtual ~Navier_Stokes() {};
			void reinit( int lev, double lin_sol_error );
			void run(	Coefficient<dim>	&coeff_function,
						RightHandSide<dim>  &RHS_function,
						BoundaryValues<dim> &bound_val_function,
						solution_type<dim>	&sol );
			/*------------------------------------------------------*/
			void run(	matrix_type<dim> 	&Stokes_operator_matrix,
						RightHandSide<dim>  &RHS_function,
						BoundaryValues<dim> &bound_val_function,
						solution_type<dim>	&sol );
			/*------------------------------------------------------*/
			void run(	Coefficient<dim>	&coeff_function,
						solution_type<dim>	&Stokes_operator_rhs,
						BoundaryValues<dim> &bound_val_function,
						solution_type<dim>	&sol );
			/*------------------------------------------------------*/
			void run(	Coefficient<dim>	&coeff_function,
						RightHandSide<dim>  &RHS_function,
						ConstraintMatrix	&constraints,
						solution_type<dim>	&sol );
			/*------------------------------------------------------*/
			void run(	matrix_type<dim> 	&Stokes_operator_matrix,
						solution_type<dim>	&Stokes_operator_rhs,
						BoundaryValues<dim> &bound_val_function,
						solution_type<dim>	&sol );
			/*------------------------------------------------------*/
			void run(	matrix_type<dim> 	&Stokes_operator_matrix,
						RightHandSide<dim>  &RHS_function,
						ConstraintMatrix	&constraints,
						solution_type<dim>	&sol );
			/*------------------------------------------------------*/
			void run(	Coefficient<dim>	&coeff_function,
						solution_type<dim>	&Stokes_operator_rhs,
						ConstraintMatrix	&constraints,
						solution_type<dim>	&sol );
			/*------------------------------------------------------*/
			void run(	matrix_type<dim> 	&Stokes_operator_matrix,
						solution_type<dim>	&Stokes_operator_rhs,
						ConstraintMatrix	&constraints,
						solution_type<dim>	&sol );
			ConstraintMatrix	constraints;
			solution_type<dim>	Stokes_operator_rhs;
			matrix_type<dim> 	Stokes_operator_matrix;
			problem_info info;

		private:
			my_timer				timer;
			int						level;
			double					sol_err;
			static FESystem<dim>	*fe;
			static QGauss<dim>		*quadrature_formula;
			static unsigned int		n_q_points;							// number of quadrature points per cell
			static unsigned int		dofs_per_cell;						// number of degrees of freedom per cell
			RightHandSide<dim>		*rhs_function;
			Coefficient<dim>		*coefficient_function;
			BoundaryValues<dim> 	*BV_function;
			matrix_type<dim> 		system_matrix;
			solution_type<dim>		system_rhs;
			solution_type<dim>		old_solution;
			std_cxx1x::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;	// preconditioner for the velocity-velocity matrix, i.e., block(0,0) in the system matrix
			int						is_newton = 0;							// Newton or simple iteration
			/*-------------------------------------------------------------------------------------------------------*/
			/*-------------------------------------------------------------------------------------------------------*/
			void assemble_Stokes_matrix( DoFHandler<dim> &dof_handler );
			void assemble_Stokes_matrix_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
														ScratchData												&common_data,
														PerTaskMatrixData										&data);
			void Stokes_matrix_copy_local_to_global (const PerTaskMatrixData &data);
			void add_convection_to_matrix (	DoFHandler<dim>		&dof_handler );
			void add_convection_to_matrix_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
														ScratchData												&common_data,
														PerTaskMatrixData										&data);
			void add_Newton_terms_to_matrix (	DoFHandler<dim>		&dof_handler );
			void add_Newton_terms_to_matrix_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
															ScratchData												&common_data,
															PerTaskMatrixData										&data);
			void system_matrix_copy_local_to_global (const PerTaskMatrixData &data);
			/*-------------------------------------------------------------------------------------------------------*/
			/*-------------------------------------------------------------------------------------------------------*/
			void assemble_Stokes_rhs( DoFHandler<dim> &dof_handler );
			void assemble_Stokes_rhs_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
													ScratchData												&common_data,
													PerTaskRhsData											&data);
			void Stokes_rhs_copy_local_to_global (const PerTaskRhsData &data);
			void add_Newton_terms_to_rhs( DoFHandler<dim> &dof_handler );
			void add_Newton_terms_to_rhs_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
														ScratchData												&common_data,
														PerTaskRhsData											&data);
			void system_rhs_copy_local_to_global (const PerTaskRhsData &data);
			/*-------------------------------------------------------------------------------------------------------*/
			/*-------------------------------------------------------------------------------------------------------*/
			void compute_constraints( DoFHandler<dim> &dof_handler );
			void apply_boundary_values( matrix_type<dim> &Stokes_operator_matrix, solution_type<dim> &Stokes_operator_rhs, ConstraintMatrix &constraints );
			/*-------------------------------------------------------------------------------------------------------*/
			/*-------------------------------------------------------------------------------------------------------*/
			void solve (	solution_type<dim> &solution,
							double lin_solver_error );
	};

	template<int dim>	FESystem<dim>*						Navier_Stokes<dim>::fe					= DiscretizationData<dim>::fe;
	template<int dim>	QGauss<dim>*						Navier_Stokes<dim>::quadrature_formula	= DiscretizationData<dim>::quadrature_formula;
	template<int dim>	unsigned int						Navier_Stokes<dim>::n_q_points			= DiscretizationData<dim>::quadrature_formula->size();
	template<int dim>	unsigned int						Navier_Stokes<dim>::dofs_per_cell		= DiscretizationData<dim>::fe->dofs_per_cell;





/*---------------------------------------------------------------------------*/
/*                           Class implementation                            */
/*---------------------------------------------------------------------------*/


	template <int dim>
	Navier_Stokes<dim>::Navier_Stokes (	int		lev,
										double	lin_sol_error )
	{
		reinit( lev, lin_sol_error );
	}


	template <int dim>
	void Navier_Stokes<dim>::reinit (	int		lev,
										double	lin_sol_error )
	{
		level	= lev;
		sol_err	= lin_sol_error;

		old_solution.reinit(level);

		Stokes_operator_matrix.reinit(level);
		Stokes_operator_rhs.reinit(level);

		system_matrix.reinit(level);
		system_rhs.reinit(level);

		A_preconditioner.reset();
	    A_preconditioner = std_cxx1x::shared_ptr<typename InnerPreconditioner<dim>::type>(new typename InnerPreconditioner<dim>::type() );
	}




	template <int dim>
	void Navier_Stokes<dim>::run(	Coefficient<dim>	&coeff_function,
									RightHandSide<dim>  &RHS_function,
									BoundaryValues<dim> &bound_val_function,
									solution_type<dim>	&sol )
	{
		info.reset();
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		coefficient_function	= &coeff_function;
		rhs_function			= &RHS_function;
		BV_function				= &bound_val_function;

		timer.tic();
			assemble_Stokes_matrix( *dof_handler );
			assemble_Stokes_rhs( *dof_handler );
			compute_constraints( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( Stokes_operator_matrix, Stokes_operator_rhs, constraints, sol );
	}


	template <int dim>
	void Navier_Stokes<dim>::run(	matrix_type<dim> 	&Stokes_operator_matrix,
									RightHandSide<dim>  &RHS_function,
									BoundaryValues<dim> &bound_val_function,
									solution_type<dim>	&sol )
	{
		info.reset();
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		rhs_function = &RHS_function;
		BV_function  = &bound_val_function;

		timer.tic();
			assemble_Stokes_rhs( *dof_handler );
			compute_constraints( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( Stokes_operator_matrix, Stokes_operator_rhs, constraints, sol );
	}


	template <int dim>
	void Navier_Stokes<dim>::run(	Coefficient<dim>	&coeff_function,
									solution_type<dim>	&Stokes_operator_rhs,
									BoundaryValues<dim> &bound_val_function,
									solution_type<dim>	&sol )
	{
		info.reset();
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		coefficient_function	= &coeff_function;
		BV_function				= &bound_val_function;

		timer.tic();
			assemble_Stokes_matrix( *dof_handler );
			compute_constraints( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( Stokes_operator_matrix, Stokes_operator_rhs, constraints, sol );
	}


	template <int dim>
	void Navier_Stokes<dim>::run(	Coefficient<dim>	&coeff_function,
									RightHandSide<dim>  &RHS_function,
									ConstraintMatrix	&constraints,
									solution_type<dim>	&sol )
	{
		info.reset();
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		coefficient_function	= &coeff_function;
		rhs_function			= &RHS_function;

		timer.tic();
			assemble_Stokes_matrix( *dof_handler );
			assemble_Stokes_rhs( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( Stokes_operator_matrix, Stokes_operator_rhs, constraints, sol );
	}


	template <int dim>
	void Navier_Stokes<dim>::run(	matrix_type<dim> 	&Stokes_operator_matrix,
									solution_type<dim>	&Stokes_operator_rhs,
									BoundaryValues<dim> &bound_val_function,
									solution_type<dim>	&sol )
	{
		info.reset();
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		BV_function = &bound_val_function;

		timer.tic();
			compute_constraints( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( Stokes_operator_matrix, Stokes_operator_rhs, constraints, sol );
	}


	template <int dim>
	void Navier_Stokes<dim>::run(	matrix_type<dim> 	&Stokes_operator_matrix,
									RightHandSide<dim>  &RHS_function,
									ConstraintMatrix	&constraints,
									solution_type<dim>	&sol )
	{
		info.reset();
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		rhs_function = &RHS_function;

		timer.tic();
			assemble_Stokes_rhs( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( Stokes_operator_matrix, Stokes_operator_rhs, constraints, sol );
	}


	template <int dim>
	void Navier_Stokes<dim>::run(	Coefficient<dim>	&coeff_function,
									solution_type<dim>	&Stokes_operator_rhs,
									ConstraintMatrix	&constraints,
									solution_type<dim>	&sol )
	{
		info.reset();
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		coefficient_function	= &coeff_function;

		timer.tic();
			assemble_Stokes_matrix( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( Stokes_operator_matrix, Stokes_operator_rhs, constraints, sol );
	}


	template <int dim>
	void Navier_Stokes<dim>::run(	matrix_type<dim> 	&Stokes_operator_matrix,
									solution_type<dim>	&Stokes_operator_rhs,
									ConstraintMatrix	&constraints,
									solution_type<dim>	&solution )
	{
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		solution_type<dim> residual_vector(level);

		double residual;
		double error = 1.0;
		int newton_steps = 0;
		int total_steps = 0;

		// remove constrained dofs (boundary values) - required to correctly calculate residual
		constraints.condense(solution.vector);

		while( error > sol_err )
		{
			system_matrix.matrix.copy_from(Stokes_operator_matrix.matrix);
			system_rhs.vector = Stokes_operator_rhs.vector;

	    	old_solution.vector = solution.vector;
	    	constraints.distribute(old_solution.vector);

	    	// add convective terms to matrix
	    	timer.tic();
	    		add_convection_to_matrix( *dof_handler );
	    	info.assemble_CPU_time += timer.toc();
	    	apply_boundary_values( system_matrix, system_rhs, constraints );

	    	// compute velocity residual
	    	system_matrix.matrix.residual( residual_vector.vector, solution.vector, system_rhs.vector );
	    	residual = residual_vector.vector.block(0).l2_norm();
	    	if ( total_steps == 0 )
	    		info.init_residual = residual;
	    	if ( residual < sol_err )
	    		break;

	    	// compute corrections to the matrix and rhs from Newton method
	    	if ( is_newton || residual < 0.1 )
	    	{
	    		timer.tic();
		    		add_Newton_terms_to_matrix( *dof_handler );
		    		add_Newton_terms_to_rhs( *dof_handler );
		    	info.assemble_CPU_time += timer.toc();
		    	apply_boundary_values( system_matrix, system_rhs, constraints );
		    	newton_steps++;
	    	}

	    	timer.tic();
	    		solve ( solution, sol_err );
	    	info.solve_CPU_time += timer.toc();

	    	// compute relative convergence of the solution
	    	constraints.condense( old_solution.vector );
	    	old_solution.vector -= solution.vector;
	    	error = old_solution.vector.l2_norm();

//	    	error = std::min(residual, error*100.0);
	    	error = residual;
	    	total_steps++;
	    	cerr << level << " " << total_steps << " " <<  residual <<  " " << error << endl;
		}
		constraints.distribute(solution.vector);

		info.total_CPU_time = info.solve_CPU_time + info.assemble_CPU_time;
		info.total_iterations = total_steps;
		info.newton_iterations = newton_steps;
	}





/*---------------------------------MATRIX------------------------------------*/
	template <int dim>
	void Navier_Stokes<dim>::assemble_Stokes_matrix( DoFHandler<dim> &dof_handler )
	{
		Stokes_operator_matrix.matrix = 0.0;

		WorkStream::run(	dof_handler.begin_active(),
							dof_handler.end(),
							*this,
							&Navier_Stokes<dim>::assemble_Stokes_matrix_on_one_cell,
							&Navier_Stokes<dim>::Stokes_matrix_copy_local_to_global,
							ScratchData(),
							PerTaskMatrixData(dofs_per_cell));
	}


	template <int dim>
	void Navier_Stokes<dim>::assemble_Stokes_matrix_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
																	ScratchData												&common_data,
																	PerTaskMatrixData										&data)
	{
		// get fe info at a current cell
		common_data.fe_values.reinit(cell);

		const FEValuesExtractors::Vector velocities(0);
		const FEValuesExtractors::Scalar pressure(dim);

	    std::vector<Tensor<2,dim> >  grad_phi_u (dofs_per_cell);
	    std::vector<double>          div_phi_u  (dofs_per_cell);
	    std::vector<double>          phi_p      (dofs_per_cell);

		// initialize local matrix
		data.cell_matrix = 0.0;

		// values of coefficient at quadrature points
		coefficient_function->vector_value_list( common_data.fe_values.get_quadrature_points(), common_data.function_values);

		// assemble local contributions
		for ( unsigned int q_point = 0; q_point < n_q_points; ++q_point )
		{
			for (unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				grad_phi_u[k] = common_data.fe_values[velocities].gradient(k, q_point);
				div_phi_u[k]  = common_data.fe_values[velocities].divergence(k, q_point);
				phi_p[k]      = common_data.fe_values[pressure].value(k, q_point);
			}
			for ( unsigned int i = 0; i < dofs_per_cell; ++i )
				for ( unsigned int j = 0; j < dofs_per_cell; ++j )
					data.cell_matrix(i,j) += (	common_data.function_values[q_point][0] * scalar_product( grad_phi_u[j], grad_phi_u[i] )
												- div_phi_u[i] * phi_p[j]
												- div_phi_u[j] * phi_p[i]
												+ phi_p[j] * phi_p[i] )
												* common_data.fe_values.JxW(q_point);
		}

		// global indexes corresponding to local nodes
		cell->get_dof_indices (data.local_dof_indices);
	}


	template <int dim>
	void Navier_Stokes<dim>::Stokes_matrix_copy_local_to_global ( const PerTaskMatrixData &data )
	{
		for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			for ( unsigned int j = 0; j < dofs_per_cell; ++j )
				Stokes_operator_matrix.matrix.add (	data.local_dof_indices[i],
													data.local_dof_indices[j],
													data.cell_matrix(i,j) );
	}

/*---------------------------------------------------------------------------*/

	template <int dim>
	void Navier_Stokes<dim>::add_convection_to_matrix( DoFHandler<dim> &dof_handler )
	{
		WorkStream::run(	dof_handler.begin_active(),
							dof_handler.end(),
							*this,
							&Navier_Stokes<dim>::add_convection_to_matrix_on_one_cell,
							&Navier_Stokes<dim>::system_matrix_copy_local_to_global,
							ScratchData(),
							PerTaskMatrixData(dofs_per_cell));
	}


	template <int dim>
	void Navier_Stokes<dim>::add_convection_to_matrix_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
																	ScratchData												&common_data,
																	PerTaskMatrixData										&data)
	{
		// get fe info at a current cell
		common_data.fe_values.reinit(cell);

		const FEValuesExtractors::Vector velocities(0);

		std::vector<Tensor<1,dim>>  phi_u      (dofs_per_cell);
	    std::vector<Tensor<2,dim>>  grad_phi_u (dofs_per_cell);

	    std::vector<Tensor<1,dim>> old_velocity_values (n_q_points);

	    common_data.fe_values[velocities].get_function_values    ( old_solution.vector, old_velocity_values    );


	    // initialize local matrix
		data.cell_matrix = 0.0;


		// assemble local contributions
		for ( unsigned int q_point = 0; q_point < n_q_points; ++q_point )
		{
			for (unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				phi_u[k]      = common_data.fe_values[velocities].value (k, q_point);
				grad_phi_u[k] = common_data.fe_values[velocities].gradient(k, q_point);
			}
			for ( unsigned int i = 0; i < dofs_per_cell; ++i )
				for ( unsigned int j = 0; j < dofs_per_cell; ++j )
					data.cell_matrix(i,j) += ( grad_phi_u[j] * old_velocity_values[q_point] ) * phi_u[i] * common_data.fe_values.JxW(q_point);
		}

		// global indexes corresponding to local nodes
		cell->get_dof_indices (data.local_dof_indices);
	}

/*---------------------------------------------------------------------------*/

	template <int dim>
	void Navier_Stokes<dim>::add_Newton_terms_to_matrix( DoFHandler<dim> &dof_handler )
	{
		WorkStream::run(	dof_handler.begin_active(),
							dof_handler.end(),
							*this,
							&Navier_Stokes<dim>::add_Newton_terms_to_matrix_on_one_cell,
							&Navier_Stokes<dim>::system_matrix_copy_local_to_global,
							ScratchData(),
							PerTaskMatrixData(dofs_per_cell));
	}


	template <int dim>
	void Navier_Stokes<dim>::add_Newton_terms_to_matrix_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
																		ScratchData												&common_data,
																		PerTaskMatrixData										&data)
	{
		// get fe info at a current cell
		common_data.fe_values.reinit(cell);

		const FEValuesExtractors::Vector velocities(0);

		std::vector<Tensor<1,dim>>  phi_u      (dofs_per_cell);
	    std::vector<Tensor<2,dim>>  grad_phi_u (dofs_per_cell);

	    std::vector<Tensor<2,dim>> old_velocity_gradients(n_q_points);

	    common_data.fe_values[velocities].get_function_gradients ( old_solution.vector, old_velocity_gradients );


	    // initialize local matrix
		data.cell_matrix = 0.0;


		// assemble local contributions
		for ( unsigned int q_point = 0; q_point < n_q_points; ++q_point )
		{
			for (unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				phi_u[k]      = common_data.fe_values[velocities].value (k, q_point);
				grad_phi_u[k] = common_data.fe_values[velocities].gradient(k, q_point);
			}
			for ( unsigned int i = 0; i < dofs_per_cell; ++i )
				for ( unsigned int j = 0; j < dofs_per_cell; ++j )
					data.cell_matrix(i,j) += ( old_velocity_gradients[q_point] * phi_u[j] ) * phi_u[i] * common_data.fe_values.JxW(q_point);
		}

		// global indexes corresponding to local nodes
		cell->get_dof_indices (data.local_dof_indices);
	}

/*---------------------------------------------------------------------------*/

	template <int dim>
	void Navier_Stokes<dim>::system_matrix_copy_local_to_global ( const PerTaskMatrixData &data )
	{
		for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			for ( unsigned int j = 0; j < dofs_per_cell; ++j )
				system_matrix.matrix.add (	data.local_dof_indices[i],
											data.local_dof_indices[j],
											data.cell_matrix(i,j) );
	}









/*------------------------------------RHS------------------------------------*/

	template <int dim>
	void Navier_Stokes<dim>::assemble_Stokes_rhs( DoFHandler<dim> &dof_handler )
	{
		Stokes_operator_rhs.vector = 0.0;

		WorkStream::run(	dof_handler.begin_active(),
							dof_handler.end(),
							*this,
							&Navier_Stokes<dim>::assemble_Stokes_rhs_on_one_cell,
							&Navier_Stokes<dim>::Stokes_rhs_copy_local_to_global,
							ScratchData(),
							PerTaskRhsData(dofs_per_cell));
	}


	template <int dim>
	void Navier_Stokes<dim>::assemble_Stokes_rhs_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
																ScratchData												&scratch_data,
																PerTaskRhsData											&private_data)
	{
		// get fe info at a current cell
		scratch_data.fe_values.reinit(cell);

		// initialize local rhs vector
		private_data.cell_rhs = 0.0;

		// values of rhs at quadrature points
		rhs_function->vector_value_list ( scratch_data.fe_values.get_quadrature_points(), scratch_data.function_values );

		// assemble local contributions
		for ( unsigned int q_point = 0; q_point < n_q_points; ++q_point )
			for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			{
				// find nonzero component of the FE vector, i.e. which variable corresponds to the current degree of freedom
				const unsigned int component_i = fe->system_to_component_index(i).first;
				private_data.cell_rhs(i) +=	scratch_data.fe_values.shape_value(i,q_point)		*
											scratch_data.function_values[q_point][component_i]	*
											scratch_data.fe_values.JxW(q_point) ;
			}

		// global indexes corresponding to local nodes
		cell->get_dof_indices ( private_data.local_dof_indices );
	}


	template <int dim>
	void Navier_Stokes<dim>::Stokes_rhs_copy_local_to_global ( const PerTaskRhsData &data )
	{
		// add local rhs  to global
		for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			Stokes_operator_rhs.vector[data.local_dof_indices[i]] += data.cell_rhs[i];
	}

/*---------------------------------------------------------------------------*/

	template <int dim>
	void Navier_Stokes<dim>::add_Newton_terms_to_rhs( DoFHandler<dim> &dof_handler )
	{
		WorkStream::run(	dof_handler.begin_active(),
							dof_handler.end(),
							*this,
							&Navier_Stokes<dim>::add_Newton_terms_to_rhs_on_one_cell,
							&Navier_Stokes<dim>::system_rhs_copy_local_to_global,
							ScratchData(),
							PerTaskRhsData(dofs_per_cell));
	}


	template <int dim>
	void Navier_Stokes<dim>::add_Newton_terms_to_rhs_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
																	ScratchData												&scratch_data,
																	PerTaskRhsData											&private_data)
	{
		// get fe info at a current cell
		scratch_data.fe_values.reinit(cell);


		const FEValuesExtractors::Vector velocities(0);

		std::vector<Tensor<1,dim>>  phi_u      (dofs_per_cell);

	    std::vector<Tensor<1,dim>> old_velocity_values (n_q_points);
	    std::vector<Tensor<2,dim>> old_velocity_gradients(n_q_points);

	    scratch_data.fe_values[velocities].get_function_values    ( old_solution.vector, old_velocity_values    );
	    scratch_data.fe_values[velocities].get_function_gradients ( old_solution.vector, old_velocity_gradients );

		// initialize local rhs vector
		private_data.cell_rhs = 0.0;


		// assemble local contributions
		for ( unsigned int q_point = 0; q_point < n_q_points; ++q_point )
			for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			{
				phi_u[i] = scratch_data.fe_values[velocities].value (i, q_point);
				private_data.cell_rhs(i) +=	( old_velocity_gradients[q_point] * old_velocity_values[q_point] ) * phi_u[i] * scratch_data.fe_values.JxW(q_point) ;
			}

		// global indexes corresponding to local nodes
		cell->get_dof_indices ( private_data.local_dof_indices );
	}


	template <int dim>
	void Navier_Stokes<dim>::system_rhs_copy_local_to_global ( const PerTaskRhsData &data )
	{
		// add local rhs  to global
		for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			system_rhs.vector[data.local_dof_indices[i]] += data.cell_rhs[i];
	}








/*-----------------------------BOUNDARY VALUES-------------------------------*/

	template <int dim>
	void Navier_Stokes<dim>::compute_constraints (	DoFHandler<dim> &dof_handler )
	{
		// boundary conditions for velocity components only
		FEValuesExtractors::Vector velocities(0);

		// compute constraints resulting from the presence of hanging nodes
		constraints.clear();
		DoFTools::make_hanging_node_constraints ( dof_handler, constraints );

		// compute Dirichlet boundary conditions
		VectorTools::interpolate_boundary_values (	dof_handler,
													0,
													*BV_function,
													constraints,
													DiscretizationData<dim>::fe->component_mask(velocities));
		VectorTools::interpolate_boundary_values (	dof_handler,
													1,
													*BV_function,
													constraints,
													DiscretizationData<dim>::fe->component_mask(velocities));
		constraints.close(); // close the filling of entries for constraint matrix
	}


	template <int dim>
	void Navier_Stokes<dim>::apply_boundary_values(	matrix_type<dim>	&system_matrix,
													solution_type<dim>	&system_rhs,
													ConstraintMatrix	&constraints )
	{
		constraints.condense( system_matrix.matrix, system_rhs.vector );
	}







/*-----------------------------SOLVE LINEAR SYSTEM---------------------------*/

	template <int dim>
	void Navier_Stokes<dim>::solve (	solution_type<dim> &solution,
										double sol_err )
	{
		system_matrix.matrix.block(1,1) = 0.0;

		solution.vector = system_rhs.vector;
		SparseDirectUMFPACK direct_solver;
		direct_solver.solve( system_matrix.matrix, solution.vector );

//		SparseMatrix<double> pressure_mass_matrix;
//		pressure_mass_matrix.reinit( DiscretizationData<dim>::sparsity_patterns_ptr[level]->block(1,1) );
//		pressure_mass_matrix.copy_from( system_matrix.matrix.block(1,1) );
//
//		// preconditioner for velocity-velocity block
//		A_preconditioner->initialize( system_matrix.matrix.block(0,0), typename InnerPreconditioner<dim>::type::AdditionalData() );
//
//		// improve solution with iterative solver
//		SparseILU<double> pmass_preconditioner;
//		pmass_preconditioner.initialize( pressure_mass_matrix, SparseILU<double>::AdditionalData() );
//		InverseMatrix<SparseMatrix<double>,SparseILU<double>> m_inverse( pressure_mass_matrix, pmass_preconditioner );
//		BlockSchurPreconditioner<typename InnerPreconditioner<dim>::type, SparseILU<double>> preconditioner( system_matrix.matrix, m_inverse, *A_preconditioner );
//
//		SolverControl solver_control( system_matrix.matrix.m()*1000, sol_err );
//
//		GrowingVectorMemory<BlockVector<double>> vector_memory;
//		SolverGMRES<BlockVector<double> >::AdditionalData gmres_data;
//		gmres_data.max_n_tmp_vectors = 100;
//		SolverGMRES<BlockVector<double>> gmres( solver_control, vector_memory, gmres_data );
//
//		gmres.solve( system_matrix.matrix, solution.vector, system_rhs.vector, preconditioner );
	}









#endif /* Navier_Stokes_H_ */
