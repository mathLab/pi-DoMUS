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


template <int dim, int spacedim, typename LAC>
BaseInterface<dim,spacedim,LAC>::
BaseInterface(const std::string &name,
              const unsigned int &ncomp,
              const unsigned int &nmat,
              const std::string &default_fe,
              const std::string &default_component_names,
              const std::string &default_differential_components) :
  ParameterAcceptor(name),
  n_components(ncomp),
  n_matrices(nmat),
  pfe(name,default_fe,default_component_names,n_components),
  str_diff_comp(default_differential_components),
  data_out("Output Parameters", "none")
{}

template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
init()
{
  matrix_couplings = std::vector<Table<2,DoFTools::Coupling> >(n_matrices);
  build_couplings();
}

template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
build_couplings()
{
  std::vector<std::string> str_couplings(n_matrices);
  set_matrix_couplings(str_couplings);

  for (unsigned int i=0; i<n_matrices; ++i)
    {
      std::vector<std::vector<unsigned int> > int_couplings;

      convert_string_to_int(str_couplings[i], int_couplings);

      matrix_couplings[i] = to_coupling(int_couplings);
    }
}


template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
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

template <int dim, int spacedim, typename LAC>
Table<2,DoFTools::Coupling>
BaseInterface<dim,spacedim,LAC>::
to_coupling(const std::vector<std::vector<unsigned int> > &coupling_table) const
{
  const unsigned int nc = n_components;
  const unsigned int nb = pfe.n_blocks();
  const std::vector<unsigned int> component_blocks = pfe.get_component_blocks();

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
        AssertThrow(coupling_table[component_blocks[i]].size() == nb,
                    ExcDimensionMismatch(coupling_table[component_blocks[i]].size(), nb));
        for (unsigned int j=0; j<nc; ++j)
          out_coupling[i][j] = m[coupling_table[component_blocks[i]][component_blocks[j]]];
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


template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
set_matrix_couplings(std::vector<std::string> &str_couplings) const
{
  std::string ones = print(std::vector<std::string>(pfe.n_blocks(),"1"));

  for (unsigned int m=0; m<n_matrices; ++m)
    {
      for (unsigned int b=0; b<pfe.n_blocks()-1; ++b)
        {
          str_couplings[m] += ones;
          str_couplings[m] += ";";
        }
      str_couplings[m] += ones;
    }
}


template <int dim, int spacedim, typename LAC>
const Table<2,DoFTools::Coupling> &
BaseInterface<dim,spacedim,LAC>::
get_matrix_coupling(const unsigned int &i) const
{
  return matrix_couplings[i];
}


template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
postprocess_newly_created_triangulation(Triangulation<dim, spacedim> &) const
{}



template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                FEValuesCache<dim,spacedim> &,
                                std::vector<SSdouble> &,
                                std::vector<std::vector<Sdouble> > &,
                                bool) const

{
  Assert(false, ExcPureFunctionCalled ());
}


template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                FEValuesCache<dim,spacedim> &,
                                std::vector<Sdouble> &,
                                std::vector<std::vector<double> > &,
                                bool) const

{
  Assert(false, ExcPureFunctionCalled ());
}


template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
compute_system_operators(const std::vector<shared_ptr<typename LADealII::BlockMatrix> >,
                         LinearOperator<typename LADealII::VectorType> &,
                         LinearOperator<typename LADealII::VectorType> &,
                         LinearOperator<typename LADealII::VectorType> &) const
{}


template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> >,
                         LinearOperator<LATrilinos::VectorType> &,
                         LinearOperator<LATrilinos::VectorType> &,
                         LinearOperator<LATrilinos::VectorType> &) const
{
  Assert(false, ExcPureFunctionCalled ());
}


template<int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
assemble_local_matrices (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         CopyData &data) const
{
  const unsigned dofs_per_cell = data.local_dof_indices.size();

  cell->get_dof_indices (data.local_dof_indices);

  std::vector<SSdouble> energies(n_matrices);
  std::vector<std::vector<Sdouble> > residuals(n_matrices,
                                               std::vector<Sdouble>(dofs_per_cell));
  assemble_energies_and_residuals(cell,
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

template<int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
assemble_local_system_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                FEValuesCache<dim,spacedim> &scratch,
                                CopyData &data) const
{
  const unsigned dofs_per_cell = data.local_dof_indices.size();

  cell->get_dof_indices (data.local_dof_indices);

  std::vector<Sdouble> energies(n_matrices);
  std::vector<std::vector<double> > residuals(n_matrices,
                                              std::vector<double>(dofs_per_cell));
  assemble_energies_and_residuals(cell,
                                  scratch,
                                  energies,
                                  residuals,
                                  true);

  for (unsigned int i=0; i<dofs_per_cell; ++i)
    residuals[0][i] += energies[0].dx(i);


  data.local_residual = residuals[0];


}



template <int dim, int spacedim, typename LAC>
std::string
BaseInterface<dim,spacedim,LAC>::get_component_names() const
{
  return pfe.get_component_names();
}


template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
BaseInterface<dim,spacedim,LAC>::get_mapping() const
{
  return StaticMappingQ1<dim,spacedim>::mapping;
}

template <int dim, int spacedim, typename LAC>
UpdateFlags
BaseInterface<dim,spacedim,LAC>::get_face_update_flags() const
{
  return (update_values         | update_quadrature_points  |
          update_normal_vectors | update_JxW_values);
}

template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &, double) const
{}

template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, Sdouble alpha) const
{
  auto &sol = fe_cache.get_current_independent_local_dofs("solution", alpha);
  auto &sol_dot = fe_cache.get_current_independent_local_dofs("solution_dot", alpha);

  for (unsigned int i=0; i<sol.size(); ++i)
    sol_dot[i] = alpha.val()*sol[i] + (sol_dot[i].val() - alpha.val()*sol[i].val());
}

template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, SSdouble alpha) const
{
  auto &sol = fe_cache.get_current_independent_local_dofs("solution", alpha);
  auto &sol_dot = fe_cache.get_current_independent_local_dofs("solution_dot", alpha);

  for (unsigned int i=0; i<sol.size(); ++i)
    sol_dot[i] = (0.5*alpha.val().val()*sol[i]) + (sol_dot[i].val().val() - alpha.val().val()*sol[i].val().val());
}

template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::initialize_data(const DoFHandler<dim,spacedim> &dof,
                                                 const typename LAC::VectorType &solution,
                                                 const typename LAC::VectorType &solution_dot,
                                                 const typename LAC::VectorType &explicit_solution,
                                                 const double t,
                                                 const double alpha) const
{
  this->dof_handler = &dof;
  this->solution = &solution;
  this->solution_dot = &solution_dot;
  this->explicit_solution = &explicit_solution;
  this->alpha = alpha;
  this->t = t;
}

template<int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::declare_parameters(ParameterHandler &prm)
{
  add_parameter(prm, &_diff_comp, "Block of differential components", str_diff_comp,
                Patterns::List(Patterns::Integer(0,1),pfe.n_blocks(),pfe.n_blocks(),","),
                "Set the blocks of differential components to 1"
                "0 for algebraic");
}

template<int dim, int spacedim, typename LAC>
const std::vector<unsigned int>
BaseInterface<dim,spacedim,LAC>::
get_differential_blocks() const
{
  return _diff_comp;
}

template<int dim, int spacedim, typename LAC>
UpdateFlags
BaseInterface<dim,spacedim,LAC>::
get_cell_update_flags() const
{
  return (update_quadrature_points |
          update_JxW_values |
          update_values |
          update_gradients);
}

template<int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
estimate_error_per_cell(const DoFHandler<dim,spacedim> &dof,
                        const typename LAC::VectorType &solution,
                        Vector<float> &estimated_error) const
{
  KellyErrorEstimator<dim,spacedim>::estimate (get_mapping(),
                                               dof,
                                               QGauss <dim-1> (dof.get_fe().degree + 1),
                                               typename FunctionMap<spacedim>::type(),
                                               solution,
                                               estimated_error,
                                               ComponentMask(),
                                               0,
                                               0,
                                               dof.get_triangulation().locally_owned_subdomain());
}

template<int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
solution_preprocessing (FEValuesCache<dim,spacedim> & /*scratch*/) const
{}

template<int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
output_solution (const unsigned int &current_cycle,
                 const unsigned int &step_number) const
{
  std::stringstream suffix;
  suffix << "." << current_cycle << "." << step_number;
  data_out.prepare_data_output( *this->dof_handler,
                                suffix.str());
  data_out.add_data_vector (*solution, get_component_names());
  std::vector<std::string> sol_dot_names =
    Utilities::split_string_list(get_component_names());
  for (auto &name : sol_dot_names)
    {
      name += "_dot";
    }
  data_out.add_data_vector (*solution_dot, print(sol_dot_names, ","));

  data_out.write_data_and_clear(get_mapping());

}

template class BaseInterface<2, 2, LATrilinos>;
template class BaseInterface<2, 3, LATrilinos>;
template class BaseInterface<3, 3, LATrilinos>;

template class BaseInterface<2, 2, LADealII>;
template class BaseInterface<2, 3, LADealII>;
template class BaseInterface<3, 3, LADealII>;
