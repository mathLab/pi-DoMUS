// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2015 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------



// Test whether TrilinosWrappers::SparseMatrix::vmult gives same result with
// Trilinos vector and parallel distributed vector

#include "tests.h"
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/parallel_vector.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>

#include "parsed_function.h"
#include "parsed_data_out.h"
#include "parameter_acceptor.h"

#include "navier_stokes.h"
#include "utilities.h"

#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

void test ()
{

  deallog.depth_console (0);
  //initlog();

  std::string parameter_filename;
  // if (argc>=2)
  // parameter_filename = argv[1];
  // else
  parameter_filename = "parameters.prm";


  const int dim = 2;
  // BoussinesqFlowProblem<dim>::Parameters  parameters(parameter_filename);

  NavierStokes<dim> flow_problem (NavierStokes<dim>::global_refinement);
  ParameterAcceptor::initialize(parameter_filename, "used_parameters.prm");


  // ParameterAcceptor::initialize("params.prm");
  //ParameterAcceptor::clear();
  // ParameterAcceptor::prm.log_parameters(deallog);

  // flow_problem.test();


}


int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
  MPILogInitAll init;
  test();

}

template class NavierStokes<2>;
