/*! \addtogroup equations
 *  @{
 */

#ifndef _neo_hookean_two_fields_interface_h_
#define _neo_hookean_two_fields_interface_h_

#include "interface.h"
#include <deal2lkit/parsed_function.h>


#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>


template <int dim, int spacedim, typename LAC>
class NeoHookeanTwoFieldsInterface : public Interface<dim,spacedim, NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>, LAC>
{
public:

  /* specific and useful functions for this very problem */
  NeoHookeanTwoFieldsInterface();

  virtual void declare_parameters (ParameterHandler &prm);

  void set_matrix_couplings(std::vector<std::string> &couplings) const;

  /* these functions MUST have the follwowing names
   *  because they are called by the ConservativeInterface class
   */
  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  /* void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                const std::vector<shared_ptr<LATrilinos::BlockMatrix> >,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const; */


private:
  double mu;

};

template <int dim, int spacedim, typename LAC>
NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>::NeoHookeanTwoFieldsInterface() :
  Interface<dim,spacedim,NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>, LAC >("NeoHookean Interface",
      dim+1,1,
      "FESystem[FE_Q(2)^d-FE_Q(1)]",
      "u,u,u,p",
      "1,0")
{
  this->init();
};



template <int dim, int spacedim,typename LAC>
template<typename EnergyType,typename ResidualType>
void NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>::energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    FEValuesCache<dim,spacedim> &fe_cache,
    std::vector<EnergyType> &energies,
    std::vector<std::vector<ResidualType> > &local_residuals,
    bool compute_only_system_terms) const
{
  EnergyType alpha = this->alpha;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);
  auto &us = fe_cache.get_values("solution", "u", displacement, alpha);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", displacement, alpha);
  auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, alpha);

  const FEValuesExtractors::Scalar pressure(dim);
  auto &ps = fe_cache.get_values("solution","p", pressure, alpha);

  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_q_points = ps.size();

  energies[0] = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      Tensor <1, dim, double> B;

      const Tensor <1, dim, EnergyType> &u = us[q];
      const Tensor <1, dim, EnergyType> &u_dot = us_dot[q];
      const EnergyType &p = ps[q];
      const Tensor <2, dim, EnergyType> &F = Fs[q];
      const Tensor<2, dim, EnergyType> C = transpose(F)*F;

      EnergyType Ic = trace(C);
      EnergyType J = determinant(F);

      EnergyType psi = (mu/2.)*(Ic-dim)-p*(J-1.);
      energies[0] += (psi)*JxW[q];
    }
}


template <int dim, int spacedim, typename LAC>
void NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  Interface<dim,spacedim, NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>,LAC >::declare_parameters(prm);
  this->add_parameter(prm, &mu, "Shear modulus", "10.0", Patterns::Double(0.0));
}

template <int dim, int spacedim, typename LAC>
void NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>::set_matrix_couplings(std::vector<std::string> &couplings) const
{
  couplings[0]="1,1;1,0";
}

template class NeoHookeanTwoFieldsInterface <3,3,LADealII>;

#endif
/*! @} */
