#include "interfaces/interface.h"
#include "lac/lac_type.h"
#include "data/assembly.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/sacado_product_type.h>

#include <deal2lkit/dof_utilities.h>
#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/any_data.h>
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parsed_mapped_functions.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>
#include <deal2lkit/utilities.h>

using namespace dealii;
using namespace deal2lkit;

template<int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::
set_time (const double &t) const
{
  dirichlet_bcs.set_time(t);
  forcing_terms.set_time(t);
  neumann_bcs.set_time(t);
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::
postprocess_newly_created_triangulation(Triangulation<dim, spacedim> &) const
{}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::
apply_dirichlet_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                     ConstraintMatrix &constraints) const
{
  if (this->operator()()->has_support_points())
    dirichlet_bcs.interpolate_boundary_values(*get_mapping(),dof_handler,constraints);
  else
    {
      const QGauss<dim-1> quad(this->operator()()->degree+1);
      dirichlet_bcs.project_boundary_values(*get_mapping(),dof_handler,quad,constraints);
    }
  dirichlet_bcs.compute_nonzero_normal_flux_constraints(dof_handler,*get_mapping(),constraints);
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::
get_preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                          Scratch &,
                          CopySystem &,
                          Sdouble &) const
{
  Assert(false, ExcPureFunctionCalled ());
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::get_preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
    Scratch &,
    CopyPreconditioner &,
    SSdouble &)  const
{
  Assert(false, ExcPureFunctionCalled ());
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::get_system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                                            Scratch &,
                                                            CopySystem &,
                                                            Sdouble &) const
{
  Assert(false, ExcPureFunctionCalled ());
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::get_system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                                            Scratch &,
                                                            CopySystem &,
                                                            SSdouble &)  const
{
  Assert(false, ExcPureFunctionCalled ());
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::get_system_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopySystem &data,
    std::vector<Sdouble> &local_residual) const
{
  SSdouble energy;
  get_system_energy(cell, scratch, data, energy);

  for (unsigned int i=0; i<local_residual.size(); ++i)
    {
      local_residual[i] = energy.dx(i);
    }
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::get_system_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopySystem &data,
    std::vector<double> &local_residual) const
{
  Sdouble energy;
  get_system_energy(cell, scratch, data, energy);

  for (unsigned int i=0; i<local_residual.size(); ++i)
    {
      local_residual[i] = energy.dx(i);
    }
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::get_preconditioner_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopyPreconditioner &data,
    std::vector<Sdouble> &local_residual) const
{
  SSdouble energy;
  get_preconditioner_energy(cell, scratch, data, energy);
  for (unsigned int i=0; i<local_residual.size(); ++i)
    {
      local_residual[i] = energy.dx(i);
    }
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::compute_system_operators(const DoFHandler<dim,spacedim> &,
    const typename LAC::BlockMatrix &,
    const typename LAC::BlockMatrix &,
    const std::vector<shared_ptr<typename LAC::BlockMatrix> >,
    LinearOperator<typename LAC::VectorType> &,
    LinearOperator<typename LAC::VectorType> &) const
{
  Assert(false, ExcPureFunctionCalled ());
}


template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::assemble_local_system (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopySystem &data) const
{
  const unsigned dofs_per_cell = data.local_dof_indices.size();

  cell->get_dof_indices (data.local_dof_indices);
  data.local_matrix = 0;

  get_system_residual(cell, scratch, data, data.sacado_residual);

  for (unsigned int i=0; i<dofs_per_cell; ++i)
    for (unsigned int j=0; j<dofs_per_cell; ++j)
      data.local_matrix(i,j) = data.sacado_residual[i].dx(j);
}


template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::assemble_local_preconditioner (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopyPreconditioner &data) const
{
  const unsigned dofs_per_cell = data.local_dof_indices.size();
  cell->get_dof_indices (data.local_dof_indices);

  data.local_matrix = 0;

  get_preconditioner_residual(cell, scratch, data, data.sacado_residual);

  for (unsigned int i=0; i<dofs_per_cell; ++i)
    for (unsigned int j=0; j<dofs_per_cell; ++j)
      data.local_matrix(i,j) = data.sacado_residual[i].dx(j);

}


template <int dim, int spacedim, int n_components, typename LAC>
shared_ptr<Mapping<dim,spacedim> >
Interface<dim,spacedim,n_components,LAC>::get_mapping(const DoFHandler<dim,spacedim> &,
                                                      const typename LAC::VectorType &) const
{
  return shared_ptr<Mapping<dim,spacedim> >(new MappingQ<dim,spacedim>(1));
}

template <int dim, int spacedim, int n_components, typename LAC>
UpdateFlags
Interface<dim,spacedim,n_components,LAC>::get_jacobian_flags() const
{
  return (update_quadrature_points |
          update_JxW_values |
          update_values |
          update_gradients);
}

template <int dim, int spacedim, int n_components, typename LAC>
UpdateFlags
Interface<dim,spacedim,n_components,LAC>::get_residual_flags() const
{
  return get_jacobian_flags();
}

template <int dim, int spacedim, int n_components, typename LAC>
UpdateFlags
Interface<dim,spacedim,n_components,LAC>::get_jacobian_preconditioner_flags() const
{
  return (update_JxW_values |
          update_values |
          update_gradients);
}

template <int dim, int spacedim, int n_components, typename LAC>
UpdateFlags
Interface<dim,spacedim,n_components,LAC>::get_face_flags() const
{
  return get_jacobian_flags();
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &, double) const
{}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, Sdouble alpha) const
{
  auto &sol = fe_cache.get_current_independent_local_dofs("solution", alpha);
  auto &sol_dot = fe_cache.get_current_independent_local_dofs("solution_dot", alpha);

  for (unsigned int i=0; i<sol.size(); ++i)
    sol_dot[i] = alpha.val()*sol[i] + (sol_dot[i].val() - alpha.val()*sol[i].val());
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, SSdouble alpha) const
{
  auto &sol = fe_cache.get_current_independent_local_dofs("solution", alpha);
  auto &sol_dot = fe_cache.get_current_independent_local_dofs("solution_dot", alpha);

  for (unsigned int i=0; i<sol.size(); ++i)
    sol_dot[i] = (0.5*alpha.val().val()*sol[i]) + (sol_dot[i].val().val() - alpha.val().val()*sol[i].val().val());
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::initialize_data(const typename LAC::VectorType &solution,
                                                          const typename LAC::VectorType &solution_dot,
                                                          const double t,
                                                          const double alpha) const
{
  if (this->old_t < t)
    {
      this->old_solution.reinit(solution, true);
      this->old_solution = solution;
      this->old_t = t;
    }

  this->solution = &solution;
  this->solution_dot = &solution_dot;
  this->alpha = alpha;
  this->t = t;
}

template<int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::declare_parameters(ParameterHandler &prm)
{
  ParsedFiniteElement<dim,spacedim>::declare_parameters(prm);
  this->add_parameter(prm, &_diff_comp, "Block of differential components", str_diff_comp,
                      Patterns::List(Patterns::Integer(0,1),this->n_blocks(),this->n_blocks(),","),
                      "Set the blocks of differential components to 1"
                      "0 for algebraic");
}

template<int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::
parse_parameters_call_back()
{
  ParsedFiniteElement<dim,spacedim>::parse_parameters_call_back();
}

template<int dim, int spacedim, int n_components, typename LAC>
const std::vector<unsigned int>
Interface<dim,spacedim,n_components,LAC>::
get_differential_blocks() const
{
  return _diff_comp;
}


// auxiliary matrices //////////////////////////////////////////////////////////
template<int dim, int spacedim, int n_components, typename LAC>
UpdateFlags
Interface<dim,spacedim,n_components,LAC>::
get_aux_matrix_flags(const unsigned int &i) const
{
  // TODO
  // return aux_matrix_update_flags[i];

  (void)i; // no warning
  return get_jacobian_preconditioner_flags();
}

template<int dim, int spacedim, int n_components, typename LAC>
unsigned int
Interface<dim,spacedim,n_components,LAC>::get_number_of_aux_matrices() const
{
  return 0;
}

template<int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::
assemble_local_aux_matrices (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                             Scratch &scratch,
                             CopyPreconditioner &data) const
{
  const unsigned dofs_per_cell = data.local_dof_indices.size();

  cell->get_dof_indices (data.local_dof_indices);
//  data.local_matrix = 0;
  get_aux_matrix_residuals(cell, scratch, data, data.sacado_residuals);

  for (unsigned aux=0; aux<get_number_of_aux_matrices(); ++aux)
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        data.local_matrices[aux](i,j) = data.sacado_residuals[aux][i].dx(j);
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::
get_aux_matrix_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                         Scratch &scratch,
                         CopyPreconditioner &data,
                         std::vector<std::vector<double> > &local_residuals) const
{
  Assert(false,ExcPureFunctionCalled());
}

template <int dim, int spacedim, int n_components, typename LAC>
void
Interface<dim,spacedim,n_components,LAC>::
get_aux_matrix_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                         Scratch &scratch,
                         CopyPreconditioner &data,
                         std::vector<std::vector<Sdouble> > &local_residuals) const
{
  Assert(false,ExcPureFunctionCalled());
}

template class Interface<2, 2, 1, LATrilinos>;
template class Interface<2, 2, 1, LADealII>;
template class Interface<2, 2, 2, LATrilinos>;
template class Interface<2, 2, 2, LADealII>;
template class Interface<2, 2, 3, LATrilinos>;
template class Interface<2, 2, 3, LADealII>;
template class Interface<2, 2, 4, LATrilinos>;
template class Interface<2, 2, 4, LADealII>;
template class Interface<2, 2, 5, LATrilinos>;
template class Interface<2, 2, 5, LADealII>;
template class Interface<2, 2, 6, LATrilinos>;
template class Interface<2, 2, 6, LADealII>;
template class Interface<2, 2, 7, LATrilinos>;
template class Interface<2, 2, 7, LADealII>;
template class Interface<2, 2, 8, LATrilinos>;
template class Interface<2, 2, 8, LADealII>;
template class Interface<3, 3, 1, LATrilinos>;
template class Interface<3, 3, 1, LADealII>;
template class Interface<3, 3, 2, LATrilinos>;
template class Interface<3, 3, 2, LADealII>;
template class Interface<3, 3, 3, LATrilinos>;
template class Interface<3, 3, 3, LADealII>;
template class Interface<3, 3, 4, LATrilinos>;
template class Interface<3, 3, 4, LADealII>;
template class Interface<3, 3, 5, LATrilinos>;
template class Interface<3, 3, 5, LADealII>;
template class Interface<3, 3, 6, LATrilinos>;
template class Interface<3, 3, 6, LADealII>;
template class Interface<3, 3, 7, LATrilinos>;
template class Interface<3, 3, 7, LADealII>;
template class Interface<3, 3, 8, LATrilinos>;
template class Interface<3, 3, 8, LADealII>;
