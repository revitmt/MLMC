/*
 * Stokes.h
 *
 *  Created on: Dec 13, 2014
 *      Author: viktor
 */

#ifndef Stokes_H_
#define Stokes_H_

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

#include <deal.II/dofs/dof_accessor.h>						// access information related to dofs while iterating trough cells, faces, etc --> get_dof_indices ( local_dof_indices )

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

using namespace dealii;




/*---------------------------------------------------------------------------*/
/*                           Class interface                                 */
/*---------------------------------------------------------------------------*/


	template <int dim>
	class Stokes
	{
		private:
		struct problem_info
		{
			int		iterations;
			double	assemble_CPU_time;
			double	solve_CPU_time;
			double	total_CPU_time;
			double 	init_residual;

			problem_info(){reset();};

			void reset()
			{
				iterations = 0;
				total_CPU_time = 0.0;
				assemble_CPU_time = 0.0;
				solve_CPU_time = 0.0;
			};

			void print( std::ostream &out )
			{
				out << std::scientific << std::setprecision(4);
				out << iterations
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
			Stokes() {};
			Stokes( int lev, double lin_sol_error );
			virtual ~Stokes() {};
			void reinit( int lev, double lin_sol_error );
			void run(	Coefficient<dim>	&coeff_function,
						RightHandSide<dim>  &RHS_function,
						BoundaryValues<dim> &bound_val_function,
						solution_type<dim>	&sol );
			void run(	matrix_type<dim> 	&operator_matrix,
						RightHandSide<dim>  &RHS_function,
						BoundaryValues<dim> &bound_val_function,
						solution_type<dim>	&sol );
			void run(	Coefficient<dim>	&coeff_function,
						solution_type<dim>	&operator_rhs,
						BoundaryValues<dim> &bound_val_function,
						solution_type<dim>	&sol );
			void run(	Coefficient<dim>	&coeff_function,
						RightHandSide<dim>  &RHS_function,
						ConstraintMatrix	&constraints,
						solution_type<dim>	&sol );
			void run(	matrix_type<dim> 	&operator_matrix,
						solution_type<dim>	&operator_rhs,
						BoundaryValues<dim> &bound_val_function,
						solution_type<dim>	&sol );
			void run(	matrix_type<dim> 	&operator_matrix,
						RightHandSide<dim>  &RHS_function,
						ConstraintMatrix	&constraints,
						solution_type<dim>	&sol );
			void run(	Coefficient<dim>	&coeff_function,
						solution_type<dim>	&operator_rhs,
						ConstraintMatrix	&constraints,
						solution_type<dim>	&sol );
			void run(	matrix_type<dim> 	&operator_matrix,
						solution_type<dim>	&operator_rhs,
						ConstraintMatrix	&constraints,
						solution_type<dim>	&sol );
			ConstraintMatrix	constraints;
			solution_type<dim>	operator_rhs;
			matrix_type<dim> 	operator_matrix;
			problem_info info;

		private:
			my_timer				timer;
			int						level;
			double					lin_sol_err;
			static FESystem<dim>	*fe;
			static QGauss<dim>		*quadrature_formula;
			static unsigned int		n_q_points;							// number of quadrature points per cell
			static unsigned int		dofs_per_cell;						// number of degrees of freedom per cell
			RightHandSide<dim>		*rhs_function;
			Coefficient<dim>		*coefficient_function;
			BoundaryValues<dim> 	*BV_function;
			matrix_type<dim> 		system_matrix;
			solution_type<dim>		system_rhs;
			/*-------------------------------------------------------------------------------------------------------*/
			/*-------------------------------------------------------------------------------------------------------*/
			void assemble_rhs( DoFHandler<dim> &dof_handler );
			void assemble_rhs_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
											ScratchData												&common_data,
											PerTaskRhsData											&data);
			void rhs_copy_local_to_global (const PerTaskRhsData &data);
			/*-------------------------------------------------------------------------------------------------------*/
			/*-------------------------------------------------------------------------------------------------------*/
			void assemble_matrix (	DoFHandler<dim>		&dof_handler );
			void assemble_matrix_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
												ScratchData												&common_data,
												PerTaskMatrixData										&data);
			void matrix_copy_local_to_global (const PerTaskMatrixData &data);
			/*-------------------------------------------------------------------------------------------------------*/
			/*-------------------------------------------------------------------------------------------------------*/
			void compute_constraints( DoFHandler<dim> &dof_handler );
			void apply_boundary_values( matrix_type<dim> &operator_matrix, solution_type<dim> &operator_rhs, ConstraintMatrix &constraints );
			/*-------------------------------------------------------------------------------------------------------*/
			/*-------------------------------------------------------------------------------------------------------*/
			void solve (	solution_type<dim> &solution,
							double lin_solver_error );
	};

	template<int dim>	FESystem<dim>*						Stokes<dim>::fe					= DiscretizationData<dim>::fe;
	template<int dim>	QGauss<dim>*						Stokes<dim>::quadrature_formula	= DiscretizationData<dim>::quadrature_formula;
	template<int dim>	unsigned int						Stokes<dim>::n_q_points			= DiscretizationData<dim>::quadrature_formula->size();
	template<int dim>	unsigned int						Stokes<dim>::dofs_per_cell		= DiscretizationData<dim>::fe->dofs_per_cell;



/*---------------------------------------------------------------------------*/
/*                           Class implementation                            */
/*---------------------------------------------------------------------------*/


	template <int dim>
	Stokes<dim>::Stokes (	int					lev,
							double				lin_sol_error )
	{
		reinit( lev, lin_sol_error );
	}


	template <int dim>
	void Stokes<dim>::reinit (	int					lev,
								double				lin_sol_error )
	{
		level		= lev;
		lin_sol_err	= lin_sol_error;

		operator_matrix.reinit(level);
		operator_rhs.reinit(level);

		system_matrix.reinit(level);
		system_rhs.reinit(level);
	}



	template <int dim>
	void Stokes<dim>::run(	Coefficient<dim>	&coeff_function,
							RightHandSide<dim>  &RHS_function,
							BoundaryValues<dim> &bound_val_function,
							solution_type<dim>	&sol )
	{
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		coefficient_function	= &coeff_function;
		rhs_function			= &RHS_function;
		BV_function				= &bound_val_function;

		timer.tic();
			assemble_matrix( *dof_handler );
			assemble_rhs( *dof_handler );
			compute_constraints( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( operator_matrix, operator_rhs, constraints, sol );
	}


	template <int dim>
	void Stokes<dim>::run(	matrix_type<dim> 	&operator_matrix,
							RightHandSide<dim>  &RHS_function,
							BoundaryValues<dim> &bound_val_function,
							solution_type<dim>	&sol )
	{
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		rhs_function			= &RHS_function;
		BV_function				= &bound_val_function;

		timer.tic();
			assemble_rhs( *dof_handler );
			compute_constraints( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( operator_matrix, operator_rhs, constraints, sol );
	}


	template <int dim>
	void Stokes<dim>::run(	Coefficient<dim>	&coeff_function,
							solution_type<dim>	&operator_rhs,
							BoundaryValues<dim> &bound_val_function,
							solution_type<dim>	&sol )
	{
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		coefficient_function	= &coeff_function;
		BV_function				= &bound_val_function;

		timer.tic();
			assemble_matrix( *dof_handler );
			compute_constraints( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( operator_matrix, operator_rhs, constraints, sol );
	}


	template <int dim>
	void Stokes<dim>::run(	Coefficient<dim>	&coeff_function,
							RightHandSide<dim>  &RHS_function,
							ConstraintMatrix	&constraints,
							solution_type<dim>	&sol )
	{
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		coefficient_function	= &coeff_function;
		rhs_function			= &RHS_function;

		timer.tic();
			assemble_matrix( *dof_handler );
			assemble_rhs( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( operator_matrix, operator_rhs, constraints, sol );
	}


	template <int dim>
	void Stokes<dim>::run(	matrix_type<dim> 	&operator_matrix,
							solution_type<dim>	&operator_rhs,
							BoundaryValues<dim> &bound_val_function,
							solution_type<dim>	&sol )
	{
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		BV_function				= &bound_val_function;

		timer.tic();
			compute_constraints( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( operator_matrix, operator_rhs, constraints, sol );
	}


	template <int dim>
	void Stokes<dim>::run(	matrix_type<dim> 	&operator_matrix,
							RightHandSide<dim>  &RHS_function,
							ConstraintMatrix	&constraints,
							solution_type<dim>	&sol )
	{
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		rhs_function			= &RHS_function;

		timer.tic();
			assemble_rhs( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( operator_matrix, operator_rhs, constraints, sol );
	}


	template <int dim>
	void Stokes<dim>::run(	Coefficient<dim>	&coeff_function,
							solution_type<dim>	&operator_rhs,
							ConstraintMatrix	&constraints,
							solution_type<dim>	&sol )
	{
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		coefficient_function	= &coeff_function;

		timer.tic();
			assemble_matrix( *dof_handler );
		info.assemble_CPU_time = timer.toc();

		run( operator_matrix, operator_rhs, constraints, sol );
	}


	template <int dim>
	void Stokes<dim>::run(	matrix_type<dim> 	&operator_matrix,
							solution_type<dim>	&operator_rhs,
							ConstraintMatrix	&constraints,
							solution_type<dim>	&solution )
	{
		apply_boundary_values( operator_matrix, operator_rhs, constraints );

		timer.tic();
			solve ( solution, lin_sol_err );
		info.solve_CPU_time = timer.toc();

		constraints.distribute(solution.vector);

		info.total_CPU_time = info.solve_CPU_time + info.assemble_CPU_time;
	}





/*---------------------------------MATRIX------------------------------------*/

	template <int dim>
	void Stokes<dim>::assemble_matrix( DoFHandler<dim> &dof_handler )
	{
		operator_matrix.matrix = 0.0;

		WorkStream::run(	dof_handler.begin_active(),
							dof_handler.end(),
							*this,
							&Stokes<dim>::assemble_matrix_on_one_cell,
							&Stokes<dim>::matrix_copy_local_to_global,
							ScratchData(),
							PerTaskMatrixData(dofs_per_cell));
	}


	template <int dim>
	void Stokes<dim>::assemble_matrix_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
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
	void Stokes<dim>::matrix_copy_local_to_global ( const PerTaskMatrixData &data )
	{
		for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			for ( unsigned int j = 0; j < dofs_per_cell; ++j )
				operator_matrix.matrix.add (	data.local_dof_indices[i],
												data.local_dof_indices[j],
												data.cell_matrix(i,j) );
	}








/*------------------------------------RHS------------------------------------*/

	template <int dim>
	void Stokes<dim>::assemble_rhs( DoFHandler<dim> &dof_handler )
	{
		operator_rhs.vector = 0.0;

		WorkStream::run(	dof_handler.begin_active(),
							dof_handler.end(),
							*this,
							&Stokes<dim>::assemble_rhs_on_one_cell,
							&Stokes<dim>::rhs_copy_local_to_global,
							ScratchData(),
							PerTaskRhsData(dofs_per_cell));
	}


	template <int dim>
	void Stokes<dim>::assemble_rhs_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
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
	void Stokes<dim>::rhs_copy_local_to_global ( const PerTaskRhsData &data )
	{
		// add local rhs  to global
		for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			operator_rhs.vector[data.local_dof_indices[i]] += data.cell_rhs[i];
	}







/*-----------------------------BOUNDARY VALUES-------------------------------*/

	template <int dim>
	void Stokes<dim>::compute_constraints (	DoFHandler<dim> &dof_handler )
	{
		// boundary conditions for velocity components only
		FEValuesExtractors::Vector velocities(0);

		// compute constraints resulting from the presence of hanging nodes
		constraints.clear();
		DoFTools::make_hanging_node_constraints ( dof_handler, constraints );

		// compute Dirichlet boundary conditions
//		for (int i = 1; i<=2; i++)
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
//		VectorTools::interpolate_boundary_values (	dof_handler,
//													2,
//													*BV_function,
//													constraints,
//													DiscretizationData<dim>::fe->component_mask(velocities));

		constraints.close(); // close the filling of entries for constraint matrix
	}


	template <int dim>
	void Stokes<dim>::apply_boundary_values( matrix_type<dim>	&operator_matrix,
											 solution_type<dim>	&operator_rhs,
											 ConstraintMatrix	&constraints )
	{
		system_matrix.matrix.copy_from(operator_matrix.matrix);
		system_rhs = operator_rhs;
		constraints.condense( system_matrix.matrix, system_rhs.vector );
	}








/*-----------------------------SOLVE LINEAR SYSTEM---------------------------*/

	template <int dim>
	void Stokes<dim>::solve (	solution_type<dim> &solution,
								double lin_sol_err )
	{
		system_matrix.matrix.block(1,1) = 0.0;

		solution = system_rhs;
		SparseDirectUMFPACK direct_solver;
		direct_solver.solve( system_matrix.matrix, solution.vector );


//		SolverControl		solver_control (10000, lin_sol_err );
//		SolverCG<>			solver (solver_control);
//		PreconditionSSOR<>	preconditioner;
//
//		preconditioner.initialize(system_matrix.matrix, 1.2);
//
////		solver.solve (system_matrix.matrix, solution.vector, rhs[level]->vector, PreconditionIdentity());
//		solver.solve (system_matrix.matrix, solution.vector, rhs[level]->vector, preconditioner);

//		iterations = solver_control.last_step();
	}









#endif /* Stokes_H_ */
