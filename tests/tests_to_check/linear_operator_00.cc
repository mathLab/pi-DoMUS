// ---------------------------------------------------------------------
//
// Copyright (C) 2015 by the deal.II authors
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

// Test that it is possible to instantiate a LinearOperator object for all
// different kinds of Trilinos matrices and vectors
// TODO: A bit more tests...

#include "tests.h"
#include "navier_stokes.h"
#include <vector>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/matrix_tools.h>

using namespace dealii;

int main(int argc, char *argv[])
{
  std::ofstream logfile("output");
  deallog.attach(logfile);
  deallog.depth_console(0);

  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, numbers::invalid_unsigned_int);


  /*------------------- Initialization : -------------------*/

  // Triangulation
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube (triangulation);
  triangulation.refine_global (2);

  // Dof Handler
  DoFHandler<2> dof_handler (triangulation);
  static const FE_Q<2 > finite_element_base(1);
  static const FESystem<2,2> finite_element(finite_element_base, 1, finite_element_base, 1);


  /*------------------- Block Matrix : -------------------*/

  // initialize the fist elements to zero and the second to 1
  std::vector<unsigned int> test_sub_blocks (2);

  test_sub_blocks[0] = 0;
  test_sub_blocks[1] = 1;


  dof_handler.distribute_dofs (finite_element);
  DoFRenumbering::component_wise (dof_handler, test_sub_blocks);

  std::vector<types::global_dof_index> dofs_per_block (2);

  DoFTools::count_dofs_per_block (dof_handler,
                                  dofs_per_block,
                                  test_sub_blocks);

  const unsigned int n_u = dofs_per_block[0],
                     n_p = dofs_per_block[1];

  TrilinosWrappers::BlockSparseMatrix       block_sparse_matrix;
  ConstraintMatrix                          test_constraints;

  std::vector<IndexSet> test_partitioning, test_relevant_partitioning;
  IndexSet test_relevant_set;
  {
    IndexSet test_index_set = dof_handler.locally_owned_dofs();
    test_partitioning.push_back(test_index_set.get_view(0,n_u));
    test_partitioning.push_back(test_index_set.get_view(n_u,n_u+n_p));

    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             test_relevant_set);
    test_relevant_partitioning.push_back(test_relevant_set.get_view(0,n_u));
    test_relevant_partitioning.push_back(test_relevant_set.get_view(n_u,n_u+n_p));

  }

  {

    FEValuesExtractors::Vector velocity_components(0);
    //boundary_conditions.set_time(time_step*time_step_number);
    // VectorTools::interpolate_boundary_values (dof_handler,
    //                                           0,
    //                                           boundary_conditions,
    //                                           test_constraints,
    //                                           finite_element.component_mask(velocity_components));

    test_constraints.close ();
  }

  block_sparse_matrix.clear ();


  TrilinosWrappers::BlockSparsityPattern sp(test_partitioning,
                                            test_partitioning,
                                            test_relevant_partitioning,
                                            MPI_COMM_WORLD);

  Table<2,DoFTools::Coupling> coupling (2, 2);
  for (unsigned int c=0; c<3; ++c)
    for (unsigned int d=0; d<3; ++d)
      if (! ((c==2) && (d==2)))
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::none;

  DoFTools::make_sparsity_pattern (dof_handler,
                                   coupling, sp,
                                   test_constraints, false,
                                   Utilities::MPI::
                                   this_mpi_process(MPI_COMM_WORLD));
  sp.compress();
  block_sparse_matrix.reinit (sp);

//

//

//
// setup_test_matrix (test_partitioning, test_relevant_partitioning);
// setup_test_preconditioner (test_partitioning,
//                                     test_relevant_partitioning);
//
// test_rhs.reinit (test_partitioning, test_relevant_partitioning,
//                           MPI_COMM_WORLD, true);
// test_solution.reinit (test_relevant_partitioning, MPI_COMM_WORLD);
// old_test_solution.reinit (test_solution);

  return 0;
}
