/*! \addtogroup equations
 *  @{
 */

#ifndef _cahn_hilliard_h
#define _cahn_hilliard_h

#include "pde_system_interface.h"
#include <deal2lkit/sacado_tools.h>




template <int dim, typename LAC>
class CahnHilliard : public PDESystemInterface<dim, dim, CahnHilliard<dim,LAC>, LAC>
{
public:

  ~CahnHilliard() {}

  CahnHilliard();

  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim>::active_cell_iterator &cell,
                              FEValuesCache<dim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


private:
  const double lambda = 1.0e-2;

};

template <int dim, typename LAC>
CahnHilliard<dim,LAC>::CahnHilliard() :
  PDESystemInterface<dim,dim,CahnHilliard<dim,LAC>, LAC>(
    "Cahn-Hilliard problem",
    2,1,
    "FESystem[FE_Q(1)-FE_Q(1)]",
    "c,mu","1,0"
  )
{}

template <int dim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
CahnHilliard<dim,LAC>::
energies_and_residuals(const typename DoFHandler<dim>::active_cell_iterator &cell,
                       FEValuesCache<dim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &local_residuals,
                       bool ) const
{
  ResidualType alpha = 0;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Scalar concentration(0);
  const FEValuesExtractors::Scalar aux(1);


  auto &cs_dot = fe_cache.get_values("solution_dot", "c", concentration, alpha);
  auto &cs = fe_cache.get_values("solution", "c", concentration, alpha);
  auto &grad_cs = fe_cache.get_gradients("solution", "c", concentration, alpha);

  auto &mus = fe_cache.get_values("solution", "mu", aux, alpha);
  auto &grad_mus = fe_cache.get_gradients("solution", "mu", aux, alpha);


  const unsigned int n_q_points = cs.size();

  auto &JxW = fe_cache.get_JxW_values();
  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const ResidualType &c = cs[q];
      const ResidualType &mu = mus[q];

      const Tensor<1,dim,ResidualType> &grad_c = grad_cs[q];
      const Tensor<1,dim,ResidualType> &grad_mu = grad_mus[q];


      const ResidualType &c_dot = cs_dot[q];

      // f = 100*c^2*(1-c)^2
      // f_prime = df/dc
      ResidualType f_prime = 200.*(c-1.)*(c-1.)*c + 200.*(c-1.)*c*c;

      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
          auto test_c = fev[concentration].value(i,q);
          auto grad_test_c = fev[concentration].gradient(i,q);

          auto test_mu = fev[aux].value(i,q);
          auto grad_test_mu = fev[aux].gradient(i,q);
          local_residuals[0][i] += (
                                     c_dot*test_c
                                     +
                                     SacadoTools::scalar_product(grad_mu,grad_test_c)
                                     +
                                     mu*test_mu
                                     -
                                     f_prime*test_mu
                                     -
                                     lambda*SacadoTools::scalar_product(grad_c,grad_test_mu)
                                   )*JxW[q];
        }


    }

}


#endif
/*! @} */
