// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2015 by the deal.II authors
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
#ifndef _LAC_TYPE_H_
#define _LAC_TYPE_H_


#include <deal.II/base/config.h>


#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>

using namespace dealii;
using namespace deal2lkit;

class LADealII
{
public:
  typedef BlockVector<double> VectorType;
  typedef BlockSparseMatrix<double> BlockMatrix;
  typedef dealii::BlockSparsityPattern BlockSparsityPattern;
};


#ifdef DEAL_II_WITH_PETSC

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_block_sparse_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

/**
 * A namespace in which the wrappers to the PETSc linear algebra classes are
 * typedef'ed to generic names. There are similar namespaces
 * LinearAlgebraDealII and LinearAlgebraTrilinos for typedefs to deal.II's own
 * classes and classes that interface with Trilinos.
 */
class LAPETSc
{
public:
  /**
   * Typedef for the type used to describe vectors that consist of multiple
   * blocks.
   */
  typedef PETScWrappers::MPI::BlockVector VectorType;

  /**
   * Typedef for the type used to describe sparse matrices that consist of
   * multiple blocks.
   */
  typedef PETScWrappers::MPI::BlockSparseMatrix BlockMatrix;

  typedef dealii::BlockSparsityPattern BlockSparsityPattern;
};

#endif // DEAL_II_WITH_PETSC

#ifdef DEAL_II_WITH_TRILINOS

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_solver.h>

/**
 * A namespace in which the wrappers to the Trilinos linear algebra classes
 * are typedef'ed to generic names. There are similar namespaces
 * LinearAlgebraDealII and LinearAlgebraPETSc for typedefs to deal.II's own
 * classes and classes that interface with PETSc.
 */
class LATrilinos
{
public:
  /**
   * Typedef for the vector type used.
   */
  typedef TrilinosWrappers::MPI::BlockVector VectorType;

  /**
   * Typedef for the type used to describe sparse matrices that consist of
   * multiple blocks.
   */
  typedef TrilinosWrappers::BlockSparseMatrix BlockMatrix;

  typedef TrilinosWrappers::BlockSparsityPattern BlockSparsityPattern;
};

#endif // DEAL_II_WITH_TRILINOS



#endif
