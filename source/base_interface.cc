#include "base_interface.h"
#include "lac/lac_type.h"
#include "copy_data.h"

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
{}

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
init()
{
  n_matrices = get_number_of_matrices();
  mapping = SP(new MappingQ<dim,spacedim>(set_mapping_degree()));
  matrices_coupling = std::vector<Table<2,DoFTools::Coupling> >(n_matrices);
  build_couplings();
}

template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
build_couplings()
{
  std::vector<std::string> str_couplings(n_matrices);
  set_matrices_coupling(str_couplings);

  for (unsigned int i=0; i<n_matrices; ++i)
    {
      std::vector<std::vector<unsigned int> > int_couplings;

      convert_string_to_int(str_couplings[i], int_couplings);

      matrices_coupling[i] = to_coupling(int_couplings);
    }
}


template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
convert_string_to_int(const std::string &str_coupling,
                      std::vector<std::vector<unsigned int> > &int_coupling) const
{
  std::vector<std::string> rows = Utilities::split_string_list(str_coupling, ';');
  for (unsigned int r=0; r<rows.size(); ++r)
    {
      std::vector<std::string> str_comp = Utilities::split_string_list(rows[r], ',');
      std::vector<unsigned int> int_comp(str_comp.size());
      for (unsigned int i=0; i<str_comp.size(); ++i)
        int_comp[i] = Utilities::string_to_int(str_comp[i]);

      int_coupling.push_back(int_comp);
    }
}

template <int dim, int spacedim, int n_components, typename LAC>
Table<2,DoFTools::Coupling>
BaseInterface<dim,spacedim,n_components,LAC>::
to_coupling(const std::vector<std::vector<unsigned int> > &coupling_table) const
{
  const unsigned int nc = n_components;
  const unsigned int nb = this->n_blocks();

  Table<2,DoFTools::Coupling> out_coupling(nc, nc);

  std::vector<DoFTools::Coupling> m(3);
  m[0] = DoFTools::none;
  m[1] = DoFTools::always;
  m[2] = DoFTools::nonzero;

  if (coupling_table.size() == nc)
    for (unsigned int i=0; i<nc; ++i)
      {
        AssertThrow(coupling_table[i].size() == nc, ExcDimensionMismatch(coupling_table[i].size(), nc));
        for (unsigned int j=0; j<nc; ++j)
          out_coupling[i][j] = m[coupling_table[i][j]];
      }
  else if (coupling_table.size() == nb)
    for (unsigned int i=0; i<nc; ++i)
      {
        AssertThrow(coupling_table[this->component_blocks[i]].size() == nb,
                    ExcDimensionMismatch(coupling_table[this->component_blocks[i]].size(), nb));
        for (unsigned int j=0; j<nc; ++j)
          out_coupling[i][j] = m[coupling_table[this->component_blocks[i]][this->component_blocks[j]]];
      }
  else if (coupling_table.size() == 0)
    for (unsigned int i=0; i<nc; ++i)
      {
        for (unsigned int j=0; j<nc; ++j)
          out_coupling[i][j] = m[1];
      }
  else
    AssertThrow(false, ExcMessage("You tried to construct a coupling with the wrong number of elements."));

  return out_coupling;
}


template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
set_matrices_coupling(std::vector<std::string> &str_couplings) const
{
  std::string ones = print(std::vector<std::string>(this->n_blocks(),"1"));

  for (unsigned int m=0; m<n_matrices; ++m)
    {
      for (unsigned int b=0; b<this->n_blocks()-1; ++b)
        {
          str_couplings[m] += ones;
          str_couplings[m] += ";";
        }
      str_couplings[m] += ones;
    }
}


template <int dim, int spacedim, int n_components, typename LAC>
const Table<2,DoFTools::Coupling> &
BaseInterface<dim,spacedim,n_components,LAC>::
get_matrix_coupling(const unsigned int &i) const
{
  return matrices_coupling[i];
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
              const Tensor<1,spacedim> normal_vector = fev.normal_vector(q);

              for (unsigned int i=0; i<local_residual.size(); ++i)
                for (unsigned int c=0; c<n_components; ++c)
                  for (unsigned int s=0; s<spacedim; ++s)
                    local_residual[i] -= T[c]*normal_vector[s]*
                                         fev.shape_value_component(i,q,c)*JxW[q];

            }// end loop over quadrature points

          break;

        } // endif face->at_boundary

    }// end loop over faces

}// end function definition



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


template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
get_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                           FEValuesCache<dim,spacedim> &,
                           std::vector<SSdouble> &,
                           std::vector<std::vector<Sdouble> > &,
                           bool) const

{
  Assert(false, ExcPureFunctionCalled ());
}


template <int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
get_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                           FEValuesCache<dim,spacedim> &,
                           std::vector<Sdouble> &,
                           std::vector<std::vector<double> > &,
                           bool) const

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

  std::vector<SSdouble> energies(n_matrices);
  std::vector<std::vector<Sdouble> > residuals(n_matrices,
                                               std::vector<Sdouble>(dofs_per_cell));
  get_energies_and_residuals(cell,
                             scratch,
                             energies,
                             residuals,
                             false);

  for (unsigned n=0; n<n_matrices; ++n)
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        residuals[n][i] += energies[n].dx(i);
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          data.local_matrices[n](i,j) = residuals[n][i].dx(j);
      }
}

template<int dim, int spacedim, int n_components, typename LAC>
void
BaseInterface<dim,spacedim,n_components,LAC>::
get_local_system_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                           FEValuesCache<dim,spacedim> &scratch,
                           CopyData &data) const
{
  const unsigned dofs_per_cell = data.local_dof_indices.size();

  cell->get_dof_indices (data.local_dof_indices);

  std::vector<Sdouble> energies(n_matrices);
  std::vector<std::vector<double> > residuals(n_matrices,
                                              std::vector<double>(dofs_per_cell));
  get_energies_and_residuals(cell,
                             scratch,
                             energies,
                             residuals,
                             true);

  for (unsigned int i=0; i<dofs_per_cell; ++i)
    residuals[0][i] += energies[0].dx(i);


  data.local_residual = residuals[0];


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
  return 0;
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
