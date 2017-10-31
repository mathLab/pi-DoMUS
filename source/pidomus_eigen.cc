#include "pidomus.h"
#include "pidomus_macros.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
//
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/dofs/dof_tools.h>

#ifdef DEAL_II_WITH_ARPACK
#include <deal.II/lac/arpack_solver.h>

#ifdef DEAL_II_ARPACK_WITH_PARPACK
#include <deal.II/lac/parpack_solver.h>
#endif

#endif

#include <deal2lkit/utilities.h>


#include "lac/lac_initializer.h"

using namespace dealii;
using namespace deal2lkit;


// This file contains the implementation of the functions
// needed for the solution of an eigenvalue problem
// through Arpack/Parpack.


#ifdef DEAL_II_WITH_ARPACK

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::solve_eigenproblem()
{
  interface.set_stepper(time_stepper);

  interface.connect_to_signals();

  // set coupling. we do it here so when we do multiple cycle,
  // we do it only once
  unsigned int nc = interface.n_components;
  Table<2,DoFTools::Coupling> mass_coupling(nc,nc);

  for (unsigned int i=0; i<nc; ++i)
    for (unsigned int j=0; j<nc; ++j)
      if (i==j)
        mass_coupling[i][j] = DoFTools::always;
      else
        mass_coupling[i][j] = DoFTools::none;

  // add the mass matrix

  typename LAC::BlockSparsityPattern mass_pattern;
  typename LAC::BlockMatrix mass_matrix;

  for (current_cycle = 0; current_cycle < n_cycles; ++current_cycle)
    {
      if (current_cycle == 0)
        {
          make_grid_fe();
          setup_dofs(true);
        }
      else
        refine_mesh();

      train_constraints[0]->distribute(solution);
      constraints_dot.distribute(solution_dot);

      // initialize mass matrix

      ScopedLACInitializer initializer(dofs_per_block,
                                       partitioning,
                                       relevant_partitioning,
                                       comm);

      mass_matrix.clear();
      initializer(mass_pattern,
                  *dof_handler,
                  *train_constraints[0],
                  mass_coupling);

      mass_matrix.reinit(mass_pattern);

      setup_jacobian(current_time,solution,solution_dot,current_alpha);

      // assemble mass matrix
      const QGauss<dim> quadrature_formula(fe->degree + 1);
      const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);

      FEValuesCache<dim,spacedim> fev_cache(interface.get_fe_mapping(),
                                            *fe, quadrature_formula,
                                            interface.get_cell_update_flags(),
                                            face_quadrature_formula,
                                            interface.get_face_update_flags());


      typedef
      FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
      CellFilter;


      mass_matrix = 0;



      auto local_copy = [&]
                        (const pidomus::CopyMass & data)
      {

        this->train_constraints[0]->distribute_local_to_global (data.local_matrix,
                                                                data.local_dof_indices,
                                                                mass_matrix);
      };

      auto local_assemble = [ this ]
                            (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                             FEValuesCache<dim,spacedim> &scratch,
                             pidomus::CopyMass & data)
      {
        this->interface.assemble_local_mass_matrix(cell, scratch, data);
      };

      {
        auto _timer = computing_timer.scoped_timer ("Assemble mass matrix");


        WorkStream::
        run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                         dof_handler->begin_active()),
             CellFilter (IteratorFilters::LocallyOwnedCell(),
                         dof_handler->end()),
             local_assemble,
             local_copy,
             fev_cache,
             pidomus::CopyMass(fe->dofs_per_cell));

        mass_matrix.compress(VectorOperation::add);
      }
      // call the eigensolver

      do_solve_eigenproblem(*matrices[0],
                            mass_matrix,
                            jacobian_op,
                            jacobian_preconditioner_op,
                            jacobian_preconditioner_op_finer,
                            eigv,
                            eigval);
      {
        auto _timer = computing_timer.scoped_timer ("Save eigenvectors");

        interface.output_eigenvectors(eigv,eigval,current_cycle);
      }
    }


}


template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::do_solve_eigenproblem(const LADealII::BlockMatrix &mat,
                                                        const LADealII::BlockMatrix &mass,
                                                        const LinearOperator<LADealII::VectorType> &,
                                                        const LinearOperator<LADealII::VectorType> &,
                                                        const LinearOperator<LADealII::VectorType> &,
                                                        std::vector<LADealII::VectorType> &eigenvectors,
                                                        std::vector<std::complex<double> > &eigenvalues)
{
  auto _timer = computing_timer.scoped_timer ("Eigenproblem solver");

  eigenvectors.resize(n_eigenvalues);

  for (unsigned int i=0; i<eigenvectors.size(); ++i)
    {
      eigenvectors[i].reinit(dofs_per_block);
      eigenvectors[i] = 0;
    }
  eigenvalues.resize(eigenvectors.size());

  SolverControl solver_control (dof_handler->n_dofs(), 1e-10);
  SparseDirectUMFPACK inverse;
  inverse.initialize (mat);


  ArpackSolver::WhichEigenvalues type;

  if (which_eigenvalues == "algebraically_largest")
    type = ArpackSolver::algebraically_largest;
  if (which_eigenvalues == "algebraically_smallest")
    type = ArpackSolver::algebraically_smallest;
  if (which_eigenvalues == "largest_magnitude")
    type = ArpackSolver::largest_magnitude;
  if (which_eigenvalues == "smallest_magnitude")
    type = ArpackSolver::smallest_magnitude;
  if (which_eigenvalues == "largest_real_part")
    type = ArpackSolver::largest_real_part;
  if (which_eigenvalues == "smallest_real_part")
    type = ArpackSolver::smallest_real_part;
  if (which_eigenvalues == "largest_imaginary_part")
    type = ArpackSolver::largest_imaginary_part;
  if (which_eigenvalues == "smallest_imaginary_part")
    type = ArpackSolver::smallest_imaginary_part;
  if (which_eigenvalues == "both_end")
    type = ArpackSolver::both_ends;

  ArpackSolver::AdditionalData additional_data(n_arnoldi_vectors,
                                               type);
  ArpackSolver eigensolver (solver_control, additional_data);
  eigensolver.solve (mat,
                     mass,
                     inverse,
                     eigenvalues,
                     eigenvectors,
                     eigenvalues.size());



  for (unsigned int i=0; i<eigenvectors.size(); ++i)
    eigenvectors[i] /= eigenvectors[i].linfty_norm ();



}

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::do_solve_eigenproblem(const LATrilinos::BlockMatrix &/*mat*/,
                                                        const LATrilinos::BlockMatrix &mass,
                                                        const LinearOperator<LATrilinos::VectorType> &jac,
                                                        const LinearOperator<LATrilinos::VectorType> &jac_prec,
                                                        const LinearOperator<LATrilinos::VectorType> &/*jac_prec_fin*/,
                                                        std::vector<LATrilinos::VectorType> &eigenvectors,
                                                        std::vector<std::complex<double> > &eigenvalues)
{
  auto _timer = computing_timer.scoped_timer ("Eigenproblem solver");

#ifdef DEAL_II_ARPACK_WITH_PARPACK

  eigenvectors.resize(n_eigenvalues);

  for (unsigned int i=0; i<eigenvectors.size(); ++i)
    {
      eigenvectors[i].reinit(partitioning,comm); // no ghost
      eigenvectors[i] = 0;
    }
  eigenvalues.resize(eigenvectors.size());



  SolverControl solver_control     (dof_handler->n_dofs(), 1e-9,
                                    /*log_history*/false,
                                    /*log_results*/false);

  PrimitiveVectorMemory<LATrilinos::VectorType> mem;

  SolverFGMRES<LATrilinos::VectorType>
  solver(solver_control, mem,
         typename SolverFGMRES<LATrilinos::VectorType>::AdditionalData(max_tmp_vector, true));

  auto S_inv = inverse_operator(jac, solver, jac_prec);



  PArpackSolver<LATrilinos::VectorType>::WhichEigenvalues type;

  if (which_eigenvalues == "algebraically_largest")
    type = PArpackSolver<LATrilinos::VectorType>::algebraically_largest;
  if (which_eigenvalues == "algebraically_smallest")
    type = PArpackSolver<LATrilinos::VectorType>::algebraically_smallest;
  if (which_eigenvalues == "largest_magnitude")
    type = PArpackSolver<LATrilinos::VectorType>::largest_magnitude;
  if (which_eigenvalues == "smallest_magnitude")
    type = PArpackSolver<LATrilinos::VectorType>::smallest_magnitude;
  if (which_eigenvalues == "largest_real_part")
    type = PArpackSolver<LATrilinos::VectorType>::largest_real_part;
  if (which_eigenvalues == "smallest_real_part")
    type = PArpackSolver<LATrilinos::VectorType>::smallest_real_part;
  if (which_eigenvalues == "largest_imaginary_part")
    type = PArpackSolver<LATrilinos::VectorType>::largest_imaginary_part;
  if (which_eigenvalues == "smallest_imaginary_part")
    type = PArpackSolver<LATrilinos::VectorType>::smallest_imaginary_part;
  if (which_eigenvalues == "both_end")
    type = PArpackSolver<LATrilinos::VectorType>::both_ends;

  PArpackSolver<LATrilinos::VectorType>::AdditionalData
  additional_data(n_arnoldi_vectors,
                  type,
                  /*symmetric=*/false);

  PArpackSolver<LATrilinos::VectorType> eigensolver (solver_control,
                                                     comm,
                                                     additional_data);
  eigensolver.reinit(global_partitioning,partitioning);

  eigensolver.solve (jac,
                     mass,
                     S_inv,
                     eigenvalues,
                     eigenvectors,
                     n_eigenvalues);
#else
  AssertThrow(false,ExcMessage("Please recompile deal.II library with DEAL_II_ARPACK_WITH_PARPACK=ON."));
  // just silence warnings about unused variables
  (void)mass;
  (void)jac;
  (void)jac_prec;
  (void)eigenvectors;
  (void)eigenvalues;
#endif //DEAL_II_ARPACK_WITH_PARPACK

}


template <int dim, int spacedim, typename LAC>
const std::vector<typename LAC::VectorType> &
piDoMUS<dim, spacedim, LAC>::get_eigenvectors()
{
  return eigv;
}

template <int dim, int spacedim, typename LAC>
const std::vector<std::complex<double> > &
piDoMUS<dim, spacedim, LAC>::get_eigenvalues()
{
  return eigval;
}


#endif // DEAL_II_WITH_ARPACK

#define INSTANTIATE(dim,spacedim,LAC) \
  template class piDoMUS<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)


