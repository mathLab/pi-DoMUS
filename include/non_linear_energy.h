#ifndef _stokes_energy_h_
#define _stokes_energy_h_

#include <deal.II/fe/fe_values.h>
#include "dof_utilities.h"
#include "parsed_finite_element.h"
#include "sak_data.h"
#include "parsed_function.h"

template <int dim,int spacedim>
class StokesEnergy : public ParsedFiniteElement<dim,spacedim>
{
public:
  StokesEnergy ();

  virtual void declare_parameters (ParameterHandler &prm);

  void add_fe_data(SAKData &d,
                   const unsigned int &dofs_per_cell,
                   const unsigned int &n_q_points) const;

  template<typename VEC>
  void add_solution(SAKData &d,
                    VEC &sol) const;

  template<typename Number>
  void initialize_preconditioner_data(SAKData &d) const;

  template<typename Number>
  void initialize_system_data(SAKData &d) const;

  template<typename Number, class Scratch, class Copy, typename VEC>
  void fill_preconditioner_data(Scratch &scratch,
                                Copy    &data,
                                const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell) const;

  template<typename Number, class Scratch, class Copy, typename VEC>
  void fill_system_data(Scratch &scratch,
                        Copy    &data,
                        const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell) const;

  template<typename Number, class Scratch>
  Number get_preconditioner_energy(const Scratch &scratch) const;

  template<typename Number, class Scratch>
  Number get_system_energy(const Scratch &scratch) const;

  /**
   * Return the component names of this problem.
   */
  static std::string default_component_names();

  static const unsigned int n_comps;
private:
  double eta;
  double rho;
  double nu;
  ParsedFunction<dim,spacedim> forcing_term;
};

template<int dim, int spacedim>
StokesEnergy<dim,spacedim>::StokesEnergy () :
  ParsedFiniteElement<dim,spacedim>("Stokes problem",
                                    "FESystem[FE_Q(2)^d-FE_Q(1)]",
                                    default_component_names(),
                                    dim+1, "1,1; 1,0", "1,0; 0,1"),
  forcing_term ("Forcing function", "2.*pi^3*cos(pi*x)*cos(pi*y); 2*pi^3*sin(pi*x)*sin(pi*y)")
{};

template<int dim, int spacedim>
void StokesEnergy<dim,spacedim>::add_fe_data(SAKData &d,
                                             const unsigned int &dofs_per_cell,
                                             const unsigned int &n_q_points) const
{
  d.add_copy(dofs_per_cell, "dofs_per_cell");
  d.add_copy(n_q_points, "n_q_points");
}

template<int dim, int spacedim>
template<typename VEC>
void StokesEnergy<dim,spacedim>::add_solution(SAKData &d,
                                              VEC &sol) const
{
  d.add_ref(sol, "sol");
}

template<int dim, int spacedim>
template<typename Number>
void StokesEnergy<dim,spacedim>::initialize_preconditioner_data(SAKData &d) const
{
  std::string suffix = typeid(Number).name();
  auto &n_q_points = d.get<unsigned int >("n_q_points");
  auto &dofs_per_cell = d.get<unsigned int >("dofs_per_cell");

  std::vector<Number> independent_local_dof_values (dofs_per_cell);
  std::vector <Number> ps(n_q_points);
  std::vector <Tensor <2, spacedim, Number> > grad_us(n_q_points);

  d.add_copy(independent_local_dof_values, "independent_local_dof_values"+suffix);

  d.add_copy(grad_us, "grad_us"+suffix);
  d.add_copy(ps, "ps"+suffix);

}

template<int dim, int spacedim>
template<typename Number>
void StokesEnergy<dim,spacedim>::initialize_system_data(SAKData &d) const
{
  std::string suffix = typeid(Number).name();
  auto &n_q_points = d.get<unsigned int >("n_q_points");
  auto &dofs_per_cell = d.get<unsigned int >("dofs_per_cell");

  std::vector<Number> independent_local_dof_values (dofs_per_cell);
  std::vector <Tensor <1, spacedim, Number> > us(n_q_points);
  std::vector <Tensor <2, spacedim, Number> > sym_grad_us(n_q_points);
  std::vector <Number> div_us(n_q_points);
  std::vector <Number> ps(n_q_points);

  d.add_copy(independent_local_dof_values, "independent_local_dof_values"+suffix);
  d.add_copy(us, "us"+suffix);
  d.add_copy(ps, "ps"+suffix);
  d.add_copy(div_us, "div_us"+suffix);
  d.add_copy(sym_grad_us, "sym_grad_us"+suffix);

}

template<int dim, int spacedim>
template<typename Number, class Scratch, class Copy, typename VEC>
void StokesEnergy<dim,spacedim>::fill_preconditioner_data(Scratch &scratch,
                                                          Copy    &data,
                                                          const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell) const
{
  std::string suffix = typeid(Number).name();
  auto &sol = scratch.anydata.template get<VEC> ("sol");
  auto &independent_local_dof_values = scratch.anydata.template get<std::vector<Number> >("independent_local_dof_values"+suffix);

  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &grad_us = scratch.anydata.template get<std::vector <Tensor <2, spacedim, Number> > >("grad_us"+suffix);

  scratch.fe_values.reinit (cell);

  DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values);
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  DOFUtilities::get_grad_values(scratch.fe_values, independent_local_dof_values, velocities, grad_us);
  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, pressure, ps);

}


template<int dim, int spacedim>
template<typename Number, class Scratch, class Copy, typename VEC>
void StokesEnergy<dim,spacedim>::fill_system_data(Scratch &scratch,
                                                  Copy    &data,
                                                  const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell) const
{
  std::string suffix = typeid(Number).name();
  auto &sol = scratch.anydata.template get<VEC> ("sol");
  auto &independent_local_dof_values = scratch.anydata.template get<std::vector<Number> >("independent_local_dof_values"+suffix);
  auto &div_us = scratch.anydata.template get<std::vector <Number> >("div_us"+suffix);
  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &us = scratch.anydata.template get<std::vector <Tensor <1, spacedim, Number> > >("us"+suffix);
  auto &sym_grad_us = scratch.anydata.template get<std::vector <Tensor <2, spacedim, Number> > >("sym_grad_us"+suffix);

  scratch.fe_values.reinit (cell);

  DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values);
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, velocities, us);
  DOFUtilities::get_div_values(scratch.fe_values, independent_local_dof_values, velocities, div_us);
  DOFUtilities::get_sym_grad_values(scratch.fe_values, independent_local_dof_values, velocities, sym_grad_us);
  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, pressure, ps);

}

template<int dim, int spacedim>
template<typename Number, class Scratch>
Number StokesEnergy<dim,spacedim>::get_preconditioner_energy(const Scratch &scratch) const
{
  std::string suffix = typeid(Number).name();
  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &grad_us = scratch.anydata.template get<std::vector <Tensor <2, spacedim, Number> > >("grad_us"+suffix);

  const unsigned int n_q_points = ps.size();

  Number energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Number &p = ps[q];
      const Tensor <2, dim, Number> &grad_u = grad_us[q];

      energy += (eta*.5*scalar_product(grad_u,grad_u) +
                 (1./eta)*0.5*p*p)*scratch.fe_values.JxW(q);
    }
  return energy;

}

template<int dim, int spacedim>
template<typename Number, class Scratch>
Number StokesEnergy<dim,spacedim>::get_system_energy(const Scratch &scratch) const
{
  std::string suffix = typeid(Number).name();
  auto &us = scratch.anydata.template get<std::vector <Tensor <1, spacedim, Number> > >("us"+suffix);
  auto &div_us = scratch.anydata.template get<std::vector <Number> > ("div_us"+suffix);
  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &sym_grad_us = scratch.anydata.template get<std::vector <Tensor <2, spacedim, Number> > >("sym_grad_us"+suffix);

  const unsigned int n_q_points = ps.size();

  Number energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      Tensor <1, dim, Number> F;
      for (unsigned int d=0; d < dim; ++d)
        {
          F[d] = forcing_term.value(scratch.fe_values.quadrature_point(q),d);
        }

      const Tensor <1, dim, Number> &u = us[q];
      const Number &div_u = div_us[q];
      const Number &p = ps[q];
      const Tensor <2, dim, Number> &sym_grad_u = sym_grad_us[q];

      Number psi = (eta*scalar_product(sym_grad_u,sym_grad_u) - p*div_u);
      energy += (psi - (F*u))*scratch.fe_values.JxW(q);
    }
  return energy;

}

template<int dim, int spacedim>
std::string StokesEnergy<dim,spacedim>::default_component_names()
{
  std::vector<std::string> names(dim+1, "u");
  names[names.size()-1] = "p";
  return print(names);
}

template <int dim, int spacedim>
void StokesEnergy<dim, spacedim>::declare_parameters (ParameterHandler &prm)
{
  ParsedFiniteElement<dim,spacedim>::declare_parameters(prm);
  this->add_parameter(prm, &eta, "eta [Pa s]", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &rho, "rho [kg/m^3]", "1000.0", Patterns::Double(0.0));
  this->add_parameter(prm, &nu, "nu", "1.0", Patterns::Double(0.0));
}


template <> const unsigned int StokesEnergy<2,2>::n_comps = 3;
template <> const unsigned int StokesEnergy<3,3>::n_comps = 4;

template class StokesEnergy <2,2>;

#endif
