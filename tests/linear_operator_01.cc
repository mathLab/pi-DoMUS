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

#include "utilities.h"

#include <fstream>
#include <iostream>
#include <vector>

using namespace dealii;

void test ()
{
  const unsigned int n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const unsigned int my_id = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const unsigned int n_rows = 3;
  const unsigned int n_cols = 4;

  IndexSet row_partitioning (n_rows);
  IndexSet col_partitioning (n_cols);

  if (n_procs == 1)
    {
      row_partitioning.add_range(0, n_rows);
      col_partitioning.add_range(0, n_cols);
    }
  else if (n_procs == 2)
    {
      // row_partitioning should be { [0, 2), [2, n_rows) }
      // col_partitioning should be { [0, 2), [2, n_cols) }
      // col_relevant_set should be { [0, 3), [1, n_cols) }
      if (my_id == 0)
        {
          row_partitioning.add_range(0, 2);
          col_partitioning.add_range(0, 2);
        }
      else if (my_id == 1)
        {
          row_partitioning.add_range(2, n_rows);
          col_partitioning.add_range(2, n_cols);
        }
    }
  else
    Assert (false, ExcNotImplemented());

  TrilinosWrappers::SparsityPattern sp (row_partitioning,
                                        col_partitioning, MPI_COMM_WORLD);
  {
    sp.add (0, 0);
    sp.add (0, 2);
    sp.add (2, 3);
  }

  sp.compress();

  TrilinosWrappers::SparseMatrix A;
  A.clear ();
  A.reinit (sp);
  {
    if (my_id == 0)
      {
        A.add (0, 0, 1);
        A.add (0, 2, 1);
        A.add (2, 3, 2.0);
      }
  }

  A.compress(VectorOperation::add);

  TrilinosWrappers::MPI::Vector x, y;
  x.reinit (col_partitioning, MPI_COMM_WORLD);
  y.reinit (row_partitioning, MPI_COMM_WORLD);

  for (unsigned int i=0; i<col_partitioning.n_elements(); ++i)
    {
      const unsigned int global_index = col_partitioning.nth_index_in_set(i);
      x(global_index) = (double) global_index;
    }

  A.vmult (y, x);

  Vector<double> ylocal = x;
  deallog << "x  : " << ylocal << std::endl;

  ylocal = y;
  deallog << "A*x: " << ylocal << std::endl;

  auto S = linear_operator<TrilinosWrappers::MPI::Vector>( A );

  S.vmult (y, x);

  ylocal = y;
  deallog << "S*y: " <<  ylocal << std::endl;
}



int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
  MPILogInitAll init;

  test();

}
