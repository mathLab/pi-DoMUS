#ifndef _LAC_INITIALIZER_H_
#define _LAC_INITIALIZER_H_

// This includes all types we know of.
#include "lac/lac_type.h"
#include <deal.II/base/mpi.h>
#include <mpi.h>

/**
 * General class, used to initialize different types of Vectors, Matrices and
 * Sparsity Patterns using a common interface.
 */
class ScopedLACInitializer
{
public:
  ScopedLACInitializer(const std::vector<types::global_dof_index> &dofs_per_block,
                       const std::vector<IndexSet> &owned,
                       const std::vector<IndexSet> &relevant,
                       const MPI_Comm extcomm= MPI_COMM_WORLD):
    dofs_per_block(dofs_per_block),
    owned(owned),
    relevant(relevant),
    comm(Utilities::MPI::duplicate_communicator(extcomm))
  {}

  /**
   * Initialize a non ghosted TrilinosWrappers::MPI::BlockVector.
   */
  void operator() (TrilinosWrappers::MPI::BlockVector &v, bool fast=false)
  {
    v.reinit(owned, comm, fast);
  }


  /**
   * Initialize a ghosted TrilinosWrappers::MPI::BlockVector.
   */
  void ghosted(TrilinosWrappers::MPI::BlockVector &v, bool fast=false)
  {
    v.reinit(owned, relevant, comm, fast);
  }

  /**
   * Initialize a serial BlockVector<double>.
   */
  void operator() (BlockVector<double> &v, bool fast=false)
  {
    v.reinit(dofs_per_block, fast);
  }


  /**
   * Initiale a ghosted BlockVector<double>. Will throw an exception and
   * should not be called...
   */
  void ghosted(BlockVector<double> &, bool fast=false)
  {
    Assert(false, ExcInternalError("You tried to create a ghosted vector in a serial run."));
    (void)fast;
  }

  /**
   * Initialize a Trilinos Sparsity Pattern.
   */
  template<int dim, int spacedim>
  void operator() (TrilinosWrappers::BlockSparsityPattern &s,
                   const DoFHandler<dim, spacedim> &dh,
                   const ConstraintMatrix &cm,
                   const Table<2,DoFTools::Coupling> &coupling)
  {
    s.reinit(owned, owned, relevant, comm);
    DoFTools::make_sparsity_pattern (dh,
                                     coupling, s,
                                     cm, false,
                                     Utilities::MPI::this_mpi_process(comm));
    s.compress();
  }

  /**
   * Initialize a Deal.II Sparsity Pattern.
   */
  template<int dim, int spacedim>
  void operator() (dealii::BlockSparsityPattern &s,
                   const DoFHandler<dim, spacedim> &dh,
                   const ConstraintMatrix &cm,
                   const Table<2,DoFTools::Coupling> &coupling)
  {
    dealii::BlockDynamicSparsityPattern csp(dofs_per_block, dofs_per_block);

    DoFTools::make_sparsity_pattern (dh,
                                     coupling, csp,
                                     cm, false);
    csp.compress();
    s.copy_from(csp);
  }

private:
  /**
   * Dofs per block.
   */
  const std::vector<types::global_dof_index> &dofs_per_block;

  /**
   * Owned dofs per block.
   */
  const std::vector<IndexSet> &owned;

  /**
   * Relevant dofs per block.
   */
  const std::vector<IndexSet> &relevant;

  /**
   * MPI Communicator.
   */
  MPI_Comm comm;
};


// A few useful functions
inline void compress(TrilinosWrappers::BlockSparseMatrix &m,
                     VectorOperation::values op)
{
  m.compress(op);
}

inline void compress(dealii::BlockSparseMatrix<double> &,
                     VectorOperation::values )
{
}
#endif
