/*
 * Project.cpp
 *
 *  Created on: Dec 18, 2014
 *      Author: viktor
 */

//#include "MLMC.h"
//#include "MonteCarlo.h"

#include <iostream>
//#include <lapacke.h>

#include "Problem.h"
//#include "DiscretizationData.h"
//#include "DataTypes.h"

using namespace MonteCarlo;
using namespace std;

int main(int argc, char **argv)
{
	const int dim = 2;
//	double eps = 0.0005; //[ 0.01 0.005 0.001 0.0005 0.0001 ]
//	double cor_length = 0.05;

//	MLMC<dim> MLMC_problem( eps, cor_length );
//	MLMC_problem.run();

	deterministic_solver::Problem<dim> problem;



	return 0;
}

