#include "base_interface.h"
#include "lac/lac_type.h"
#include "copy_data.h"

#include "pidomus_macros.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
//#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/sacado_product_type.h>

#include <deal2lkit/dof_utilities.h>
#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/any_data.h>
#include <deal2lkit/utilities.h>
#include <deal2lkit/sacado_tools.h>
#include <deal2lkit/fe_values_cache.h>

using namespace dealii;
using namespace deal2lkit;
using namespace SacadoTools;


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

    std::cout << "Energy = " << SacadoTools::to_double(energies[0]) << std::endl;


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
BaseInterface<dim,spacedim,LAC>::get_default_mapping() const
{
  return StaticMappingQ1<dim,spacedim>::mapping;
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
BaseInterface<dim,spacedim,LAC>::get_output_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
BaseInterface<dim,spacedim,LAC>::get_fe_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
BaseInterface<dim,spacedim,LAC>::get_bc_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
BaseInterface<dim,spacedim,LAC>::get_kelly_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
BaseInterface<dim,spacedim,LAC>::get_error_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
BaseInterface<dim,spacedim,LAC>::get_interpolate_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
BaseInterface<dim,spacedim,LAC>::get_project_mapping() const
{
  return get_default_mapping();
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
BaseInterface<dim,spacedim,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &/*fe_cache*/,
    double /*alpha*/) const
{}

template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, Sdouble alpha) const
{
  auto &sol = fe_cache.get_current_independent_local_dofs("solution", alpha);
  auto &sol_dot = fe_cache.get_current_independent_local_dofs("solution_dot", alpha);
  double a = to_double(alpha);
  for (unsigned int i=0; i<sol.size(); ++i)
    sol_dot[i] = a*sol[i] +  (to_double(sol_dot[i]) - a*to_double(sol[i]));
}

template <int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, SSdouble alpha) const
{
  auto &sol = fe_cache.get_current_independent_local_dofs("solution", alpha);
  auto &sol_dot = fe_cache.get_current_independent_local_dofs("solution_dot", alpha);

  for (unsigned int i=0; i<sol.size(); ++i)
    sol_dot[i] = (0.5*to_double(alpha)*sol[i]) + (to_double(sol_dot[i]) - to_double(alpha)*to_double(sol[i]));
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
estimate_error_per_cell(Vector<float> &estimated_error) const
{
  const DoFHandler<dim,spacedim> &dof = this->get_dof_handler();
  KellyErrorEstimator<dim,spacedim>::estimate (get_kelly_mapping(),
                                               dof,
                                               QGauss <dim-1> (dof.get_fe().degree + 1),
                                               typename FunctionMap<spacedim>::type(),
                                               this->get_locally_relevant_solution(),
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
  data_out.prepare_data_output( this->get_dof_handler(),
                                suffix.str());
  data_out.add_data_vector (this->get_locally_relevant_solution(),
                            get_component_names());
  std::vector<std::string> sol_dot_names =
    Utilities::split_string_list(get_component_names());
  for (auto &name : sol_dot_names)
    {
      name += "_dot";
    }
  data_out.add_data_vector (this->get_locally_relevant_solution_dot(),
                            print(sol_dot_names, ","));

  data_out.write_data_and_clear(get_output_mapping());

}

#ifdef DEAL_II_WITH_ARPACK

template<int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
output_eigenvectors(const std::vector<typename LAC::VectorType> &eigenvectors,
                    const std::vector<std::complex<double> > &eigenvalues,
                    const unsigned int &current_cycle) const
{
  auto &pcout = this->get_pcout();

  for (unsigned int i=0; i<eigenvectors.size(); ++i)
    {
      pcout << "eigenvalues[" <<i<<"] = " << eigenvalues[i]<< std::endl;
      std::stringstream suffix;
      suffix << ".eig." << current_cycle << "."<<i;
      data_out.prepare_data_output(this->get_dof_handler(),suffix.str());

      std::vector<std::string> eig_names =
        Utilities::split_string_list(get_component_names());
      for (auto &name : eig_names)
        {
          name += "_";
          name += Utilities::int_to_string(i);
        }

      data_out.add_data_vector (eigenvectors[i],
                                get_component_names());

      data_out.write_data_and_clear(get_output_mapping());

    }
}


template<int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
assemble_local_mass_matrix (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                            FEValuesCache<dim,spacedim> &scratch,
                            CopyMass &data) const
{
  const unsigned dofs_per_cell = data.local_dof_indices.size();

  cell->get_dof_indices (data.local_dof_indices);

  double dummy=0;
  reinit(dummy,cell,scratch);
  auto &fev = scratch.get_current_fe_values();
  const unsigned int nq = fev.n_quadrature_points;

  data.local_matrix = 0;


  const auto &fe = fev.get_fe();

  if (fe.is_primitive())
    {
      for (unsigned int q=0; q<nq; ++q)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              if (fe.system_to_component_index(i).first == fe.system_to_component_index(j).first)
                {
                  data.local_matrix(i,j) +=
                    fev.shape_value(i,q)
                    *
                    fev.shape_value(j,q)
                    *fev.JxW(q);
                }
            }

    }
  else
    {
      for (unsigned int q=0; q<nq; ++q)
        for (unsigned int c=0; c<n_components; ++c)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              data.local_matrix(i,j) +=
                fev.shape_value_component(i,q,c)
                *
                fev.shape_value_component(j,q,c)
                *fev.JxW(q);
    }
}

#endif

template<int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::
set_stepper (const std::string &s) const
{
  stepper = s;
}


template<int dim, int spacedim, typename LAC>
void
BaseInterface<dim,spacedim,LAC>::connect_to_signals() const
{}

template<int dim, int spacedim, typename LAC>
double
BaseInterface<dim,spacedim,LAC>::
vector_norm (const typename LAC::VectorType &v) const
{
  return v.l2_norm();
}


#define INSTANTIATE(dim,spacedim,LAC) \
  template class BaseInterface<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)


