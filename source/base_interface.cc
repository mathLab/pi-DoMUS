#include "base_interface.h"
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

template <int dim, int spacedim, int n_components, typename LAC>
BaseInterface<dim,spacedim,n_components,LAC>::
  BaseInterface(const std::string &name,
                const std::string &default_fe,
                const std::string &default_component_names,
                const std::string &default_differential_components) :
    ParsedFiniteElement<dim,spacedim>(name, default_fe, default_component_names,
                                      n_components),
    forcing_terms("Forcing terms", default_component_names, ""),
    neumann_bcs("Neumann boundary conditions", default_component_names, ""),
dirichlet_bcs("Dirichlet boundary conditions", default_component_names, "0=ALL"),
dirichlet_bcs_dot("Time derivative of Dirichlet boundary conditions", default_component_names, ""),
    str_diff_comp(default_differential_components),
    old_t(-1.0)
  {
    n_matrices = get_number_of_matrices();
    mapping = SP(new MappingQ<dim,spacedim>(set_mapping_degree());
    matrices_coupling = std::vector<Table<2,DoFTools::Coupling> >(n_matrices);
    set_matrices_coupling(matrices_coupling);
  }


template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
apply_neumann_bcs (
  const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
  FEValuesCache<dim,spacedim> &scratch,
  std::vector<double> &local_residual) const
{

  double dummy = 0.0;

  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      unsigned int face_id = cell->face(face)->boundary_id();
      if (cell->face(face)->at_boundary() && neumann_bcs.acts_on_id(face_id))
        {
          this->reinit(dummy, cell, face, scratch);

          auto &fev = scratch.get_current_fe_values();
          auto &q_points = scratch.get_quadrature_points();
          auto &JxW = scratch.get_JxW_values();

          for (unsigned int q=0; q<q_points.size(); ++q)
            {
              Vector<double> T(n_components);
              neumann_bcs.get_mapped_function(face_id)->vector_value(q_points[q], T);

              for (unsigned int i=0; i<local_residual.size(); ++i)
                for (unsigned int c=0; c<n_components; ++c)
                  local_residual[i] -= T[c]*fev.normal_vector
		    fev.shape_value_component(i,q,c)*JxW[q];

            }
          break;
        }
    }
}

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
apply_forcing_terms (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                     FEValuesCache<dim,spacedim> &scratch,
                     std::vector<double> &local_residual) const
{
  unsigned cell_id = cell->material_id();
  if (forcing_terms.acts_on_id(cell_id))
    {
      double dummy = 0.0;
      this->reinit(dummy, cell, scratch);

      auto &fev = scratch.get_current_fe_values();
      auto &q_points = scratch.get_quadrature_points();
      auto &JxW = scratch.get_JxW_values();
      for (unsigned int q=0; q<q_points.size(); ++q)
        for (unsigned int i=0; i<local_residual.size(); ++i)
          for (unsigned int c=0; c<n_components; ++c)
            {
              double B = forcing_terms.get_mapped_function(cell_id)->value(q_points[q],c);
              local_residual[i] -= B*fev.shape_value_component(i,q,c)*JxW[q];
            }
    }
}



template<int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
set_time (const double &t) const
{
  dirichlet_bcs.set_time(t);
  dirichlet_bcs_dot.set_time(t);
  forcing_terms.set_time(t);
  neumann_bcs.set_time(t);
}

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
postprocess_newly_created_triangulation(Triangulation<dim, spacedim> &) const
{}


template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
apply_dirichlet_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                     ConstraintMatrix &constraints) const
{
  if (this->operator()()->has_support_points())
    dirichlet_bcs.interpolate_boundary_values(*(this->get_mapping()),dof_handler,constraints);
  else
    {
      const QGauss<dim-1> quad(this->operator()()->degree+1);
      dirichlet_bcs.project_boundary_values(*(this->get_mapping()),dof_handler,quad,constraints);
    }
  dirichlet_bcs.compute_nonzero_normal_flux_constraints(dof_handler,*(this->get_mapping()),constraints);
}

///////////////////////// energies /////////////////////////////////////////////

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
get_energies(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
             FEValuesCache<dim,spacedim> &,
             std::vector<SSdouble> &) const
{
  Assert(false, ExcPureFunctionCalled ());
}


template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
get_energies(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
             FEValuesCache<dim,spacedim> &,
             std::vector<Sdouble> &) const
{
  Assert(false, ExcPureFunctionCalled ());
}


///////////////////////// residuals ////////////////////////////////////////////

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
get_residuals (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
               FEValuesCache<dim,spacedim> &,
               std::vector<std::vector<Sdouble> > &) const
{
  Assert(false, ExcPureFunctionCalled ());
}

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
get_residuals (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
               FEValuesCache<dim,spacedim> &,
               std::vector<std::vector<double> > &) const
{
  Assert(false, ExcPureFunctionCalled ());
}




template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
compute_system_operators(const DoFHandler<dim,spacedim> &,
                         const std::vector<shared_ptr<typename LAC::BlockMatrix> >,
                         LinearOperator<typename LAC::VectorType> &,
                         LinearOperator<typename LAC::VectorType> &) const
{
  Assert(false, ExcPureFunctionCalled ());
}


template<int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
assemble_local_matrices (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         CopyData &data) const
{
  const unsigned dofs_per_cell = data.local_dof_indices.size();

  cell->get_dof_indices (data.local_dof_indices);

  std::vector<SSdouble> energies(n_aux_matrices);
  get_energies_and_residuals(cell,
                             scratch,
                             energies,
                             data.sacado_residuals,
                             false);

  for (unsigned n=0; n<n_matrices; ++n)
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        data.sacado_residuals[n][i] += energies[n].dx(i);
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          data.local_matrices[n](i,j) = data.sacado_residuals[n][i].dx(j);
      }
}


template <int dim, int spacedim, int n_components, typename LAC>
shared_ptr<Mapping<dim,spacedim> >
BaseInterface<dim,spacedim,n_components,LAC>::get_mapping() const
{
  return mapping;
}

template <int dim, int spacedim, int n_components, typename LAC>
unsigned int
BaseInterface<dim,spacedim,n_components,LAC>::set_mapping_degree() const
{
  return 1;
}

template <int dim, int spacedim, int n_components, typename LAC>
UpdateFlags
BaseInterface<dim,spacedim,n_components,LAC>::get_face_update_flags() const
{
  return (update_values         | update_quadrature_points  |
	  update_normal_vectors | update_JxW_values);
}

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &, double) const
{}

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, Sdouble alpha) const
{
  auto &sol = fe_cache.get_current_independent_local_dofs("solution", alpha);
  auto &sol_dot = fe_cache.get_current_independent_local_dofs("solution_dot", alpha);

  for (unsigned int i=0; i<sol.size(); ++i)
    sol_dot[i] = alpha.val()*sol[i] + (sol_dot[i].val() - alpha.val()*sol[i].val());
}

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, SSdouble alpha) const
{
  auto &sol = fe_cache.get_current_independent_local_dofs("solution", alpha);
  auto &sol_dot = fe_cache.get_current_independent_local_dofs("solution_dot", alpha);

  for (unsigned int i=0; i<sol.size(); ++i)
    sol_dot[i] = (0.5*alpha.val().val()*sol[i]) + (sol_dot[i].val().val() - alpha.val().val()*sol[i].val().val());
}

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::initialize_data(const typename LAC::VectorType &solution,
    const typename LAC::VectorType &solution_dot,
    const double t,
    const double alpha) const
{
  if (old_t < t)
    {
      old_solution.reinit(solution);
      old_t = t;
    }

  this->solution = &solution;
  this->solution_dot = &solution_dot;
  this->alpha = alpha;
  this->t = t;
}

template<int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::declare_parameters(ParameterHandler &prm)
{
  ParsedFiniteElement<dim,spacedim>::declare_parameters(prm);
  this->add_parameter(prm, &_diff_comp, "Block of differential components", str_diff_comp,
                      Patterns::List(Patterns::Integer(0,1),this->n_blocks(),this->n_blocks(),","),
                      "Set the blocks of differential components to 1"
                      "0 for algebraic");
}

template<int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
parse_parameters_call_back()
{
  ParsedFiniteElement<dim,spacedim>::parse_parameters_call_back();
}

template<int dim, int spacedim, int n_components, typename LAC>
const std::vector<unsigned int>
BaseInterface<dim,spacedim,n_components,LAC>::
get_differential_blocks() const
{
  return _diff_comp;
}

template<int dim, int spacedim, int n_components, typename LAC>
UpdateFlags
BaseInterface<dim,spacedim,n_components,LAC>::
get_matrices_update_flags() const
{
  return (update_quadrature_points |
          update_JxW_values |
          update_values |
          update_gradients);
}

template<int dim, int spacedim, int n_components, typename LAC>
unsigned int
BaseInterface<dim,spacedim,n_components,LAC>::get_number_of_matrices() const
{
  Assert(false,ExcPureFunctionCalled());
}

template class BaseInterface<2, 2, 1, LATrilinos>;
template class BaseInterface<2, 2, 1, LADealII>;
template class BaseInterface<2, 2, 2, LATrilinos>;
template class BaseInterface<2, 2, 2, LADealII>;
template class BaseInterface<2, 2, 3, LATrilinos>;
template class BaseInterface<2, 2, 3, LADealII>;
template class BaseInterface<2, 2, 4, LATrilinos>;
template class BaseInterface<2, 2, 4, LADealII>;
template class BaseInterface<2, 2, 5, LATrilinos>;
template class BaseInterface<2, 2, 5, LADealII>;
template class BaseInterface<2, 2, 6, LATrilinos>;
template class BaseInterface<2, 2, 6, LADealII>;
template class BaseInterface<2, 2, 7, LATrilinos>;
template class BaseInterface<2, 2, 7, LADealII>;
template class BaseInterface<2, 2, 8, LATrilinos>;
template class BaseInterface<2, 2, 8, LADealII>;
template class BaseInterface<3, 3, 1, LATrilinos>;
template class BaseInterface<3, 3, 1, LADealII>;
template class BaseInterface<3, 3, 2, LATrilinos>;
template class BaseInterface<3, 3, 2, LADealII>;
template class BaseInterface<3, 3, 3, LATrilinos>;
template class BaseInterface<3, 3, 3, LADealII>;
template class BaseInterface<3, 3, 4, LATrilinos>;
template class BaseInterface<3, 3, 4, LADealII>;
template class BaseInterface<3, 3, 5, LATrilinos>;
template class BaseInterface<3, 3, 5, LADealII>;
template class BaseInterface<3, 3, 6, LATrilinos>;
template class BaseInterface<3, 3, 6, LADealII>;
template class BaseInterface<3, 3, 7, LATrilinos>;
template class BaseInterface<3, 3, 7, LADealII>;
template class BaseInterface<3, 3, 8, LATrilinos>;
template class BaseInterface<3, 3, 8, LADealII>;
