/*
 * Output.h
 *
 *  Created on: Dec 22, 2014
 *      Author: viktor
 */

#ifndef OUTPUT_H_
#define OUTPUT_H_

#include <deal.II/numerics/data_out.h>

//#include "DiscretizationData.h"
#include "DataTypes.h"
#include <string>

using namespace MonteCarlo;
using namespace dealii;

template<int dim>
class Output
{
	public:
		Output() {};
		virtual ~Output() {};
		void print( const solution_type<dim> &solution , const std::string filename );
		void print_exact_function( const int level, Function<dim> &function_to_plot, const std::string filename );
};


template<int dim>
void Output<dim>::print( const solution_type<dim> &solution, const std::string filename )
{
	DataOut<dim> data_out;
	data_out.attach_dof_handler ( *(DiscretizationData<dim>::dof_handlers_ptr[solution.level]) );
	data_out.add_data_vector ( solution.vector, "solution" );
	data_out.build_patches ();
//
//	std::ofstream output (filename+".gpl");
//	data_out.write_gnuplot (output);

    std::ofstream output (filename+".vtk");
    data_out.write_vtk (output);
}


template<int dim>
void Output<dim>::print_exact_function( const int level, Function<dim> &function_to_plot, const std::string filename )
{
	solution_type<dim> function(level);
	VectorTools::interpolate ( *(DiscretizationData<dim>::dof_handlers_ptr[level]), function_to_plot, function.vector );

	DataOut<dim> data_out;
	data_out.attach_dof_handler ( *(DiscretizationData<dim>::dof_handlers_ptr[level]) );
	data_out.add_data_vector ( function.vector, "function" );
	data_out.build_patches ();

//	std::ofstream output (filename+".gpl");
//	data_out.write_gnuplot (output);

    std::ofstream output (filename+".vtk");
    data_out.write_vtk (output);
}


#endif /* OUTPUT_H_ */
