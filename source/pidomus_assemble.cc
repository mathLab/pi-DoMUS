#include "pidomus.h"
#include "pidomus_macros.h"
#include "copy_data.h"

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/base/work_stream.h>

using namespace dealii;
using namespace deal2lkit;


// This file contains the functions whose aim is to assemble a global
// matrix or vector relying on the WorkStream function.


template <int dim, int spacedim, typename LAC>
int
piDoMUS<dim, spacedim, LAC>::setup_jacobian(const double t,
                                            const typename LAC::VectorType &src_yy,
                                            const typename LAC::VectorType &src_yp,
                                            const double alpha)
{
  signals.begin_setup_jacobian();

  current_alpha = alpha;
  syncronize(t,solution,solution_dot);

  assemble_matrices(t, src_yy, src_yp, alpha);
  if (use_direct_solver == false)
    {
      auto _timer = computing_timer.scoped_timer ("Compute system operators");

      interface.compute_system_operators(matrices,
                                         jacobian_op,
                                         jacobian_preconditioner_op,
                                         jacobian_preconditioner_op_finer);
    }
  signals.end_setup_jacobian();
  return 0;
}



template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::assemble_matrices (const double t,
                                                     const typename LAC::VectorType &solution,
                                                     const typename LAC::VectorType &solution_dot,
                                                     const double alpha)
{
  auto _timer = computing_timer.scoped_timer ("Assemble matrices");

  signals.begin_assemble_matrices();

  current_alpha = alpha;
  syncronize(t,solution,solution_dot);

  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);

  FEValuesCache<dim,spacedim> fev_cache(interface.get_fe_mapping(),
                                        *fe, quadrature_formula,
                                        interface.get_cell_update_flags(),
                                        face_quadrature_formula,
                                        interface.get_face_update_flags());

  interface.solution_preprocessing(fev_cache);

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;


  for (unsigned int i=0; i<n_matrices; ++i)
    *(matrices[i]) = 0;



  auto local_copy = [this]
                    (const pidomus::CopyData & data)
  {

    for (unsigned int i=0; i<n_matrices; ++i)
      this->train_constraints[i]->distribute_local_to_global (data.local_matrices[i],
                                                              data.local_dof_indices,
                                                              *(this->matrices[i]));
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         pidomus::CopyData & data)
  {
    this->interface.assemble_local_matrices(cell, scratch, data);
  };



  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       fev_cache,
       pidomus::CopyData(fe->dofs_per_cell,n_matrices));

  for (unsigned int i=0; i<n_matrices; ++i)
    matrices[i]->compress(VectorOperation::add);
}




template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::get_lumped_mass_matrix(typename LAC::VectorType &dst) const
{
  auto _timer = computing_timer.scoped_timer ("Assemble lumped mass matrix");


  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);


  FEValuesCache<dim,spacedim> fev_cache(interface.get_fe_mapping(),
                                        *fe, quadrature_formula,
                                        interface.get_cell_update_flags(),
                                        face_quadrature_formula,
                                        interface.get_face_update_flags());


  dst = 0;

  auto local_copy = [&dst, this] (const pidomus::CopyData & data)
  {

    for (unsigned int i=0; i<data.local_dof_indices.size(); ++i)
      {
        // kinsol needs that each component must strictly greater
        // than zero.
        dst[ data.local_dof_indices[i] ] += data.local_residual[i] + 1e-12;
      }
  };

  auto local_assemble = []
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         pidomus::CopyData & data)
  {
    const unsigned dofs_per_cell = data.local_residual.size();
    scratch.reinit(cell);

    cell->get_dof_indices (data.local_dof_indices);

    auto &JxW = scratch.get_JxW_values();
    auto &fev = scratch.get_current_fe_values();

    const unsigned int n_q_points = JxW.size();

    std::fill(data.local_residual.begin(), data.local_residual.end(), 0.0);
    for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        data.local_residual[i] += fev.shape_value(i,q)*JxW[q];

  };

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       fev_cache,
       pidomus::CopyData(fe->dofs_per_cell,n_matrices));


  dst.compress(VectorOperation::add);

}



template <int dim, int spacedim, typename LAC>
int
piDoMUS<dim, spacedim, LAC>::residual(const double t,
                                      const typename LAC::VectorType &solution,
                                      const typename LAC::VectorType &solution_dot,
                                      typename LAC::VectorType &dst)
{
  auto _timer = computing_timer.scoped_timer ("Residual");

  signals.begin_residual();

  for (auto i : solution)
    std::cout << i<<"    ";

  std::cout << std::endl;

  syncronize(t,solution,solution_dot);

  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);


  FEValuesCache<dim,spacedim> fev_cache(interface.get_fe_mapping(),
                                        *fe, quadrature_formula,
                                        interface.get_cell_update_flags(),
                                        face_quadrature_formula,
                                        interface.get_face_update_flags());

  interface.solution_preprocessing(fev_cache);

  dst = 0;

  auto local_copy = [&dst, this] (const pidomus::CopyData & data)
  {

    this->train_constraints[0]->distribute_local_to_global (data.local_residual,
                                                            data.local_dof_indices,
                                                            dst);
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         pidomus::CopyData & data)
  {
    this->interface.assemble_local_system_residual(cell,scratch,data);
    // apply conservative loads
    this->apply_forcing_terms(cell, scratch, data.local_residual);

    if (cell->at_boundary())
      this->apply_neumann_bcs(cell, scratch, data.local_residual);


  };

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       fev_cache,
       pidomus::CopyData(fe->dofs_per_cell,n_matrices));


  dst.compress(VectorOperation::add);

  auto id = solution.locally_owned_elements();
  for (unsigned int i = 0; i < id.n_elements(); ++i)
    {
      auto j = id.nth_index_in_set(i);
      if (train_constraints[0]->is_constrained(j))
        dst[j] = solution(j) - locally_relevant_solution(j);
    }

  dst.compress(VectorOperation::insert);

  signals.end_residual();

  return 0;
}



#define INSTANTIATE(dim,spacedim,LAC) \
  template class piDoMUS<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)
