/*! \addtogroup equations
 *  @{
 */

#ifndef _neo_hookean_two_fields_interface_h_
#define _neo_hookean_two_fields_interface_h_

#include "pde_system_interface.h"
#include <deal2lkit/parsed_function.h>

#include <deal.II/lac/solver_cg.h>


template <int dim, int spacedim, typename LAC>
class NeoHookeanTwoFieldsInterface : public PDESystemInterface<dim,spacedim, NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>, LAC>
{
public:

  /* specific and useful functions for this very problem */
  NeoHookeanTwoFieldsInterface();

  virtual void declare_parameters (ParameterHandler &prm);

  void set_matrix_couplings(std::vector<std::string> &couplings) const;


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  void compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const;


private:
  double mu;

  mutable shared_ptr<TrilinosWrappers::PreconditionAMG> Amg_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> P_preconditioner;

};

template <int dim, int spacedim, typename LAC>
NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>::NeoHookeanTwoFieldsInterface() :
  PDESystemInterface<dim,spacedim,NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>, LAC >("NeoHookean Parameters",
      dim+1,2,
      "FESystem[FE_Q(2)^d-FE_Q(1)]",
      "u,u,u,p",
      "1,0")
{}


template <int dim, int spacedim,typename LAC>
template<typename EnergyType,typename ResidualType>
void NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>::energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    FEValuesCache<dim,spacedim> &fe_cache,
    std::vector<EnergyType> &energies,
    std::vector<std::vector<ResidualType> > &,
    bool compute_only_system_terms) const
{
  EnergyType alpha = 0;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);

  auto &grad_us = fe_cache.get_gradients("solution", "u", displacement, alpha);
  auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, alpha);

  const FEValuesExtractors::Scalar pressure(dim);
  auto &ps = fe_cache.get_values("solution","p", pressure, alpha);

  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_q_points = ps.size();

  energies[0] = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      Tensor <1, dim, double> B;

      const EnergyType &p = ps[q];
      const Tensor <2, dim, EnergyType> &F = Fs[q];
      const Tensor<2, dim, EnergyType> C = transpose(F)*F;
      auto &grad_u = grad_us[q];

      EnergyType Ic = trace(C);
      EnergyType J = determinant(F);

      EnergyType psi = (mu/2.)*(Ic-dim)-p*(J-1.);
      energies[0] += (psi)*JxW[q];
      if (!compute_only_system_terms)
        energies[1] += (scalar_product(grad_u,grad_u) + 0.5*p*p)*JxW[q];
    }
}


template <int dim, int spacedim, typename LAC>
void NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>,LAC >::declare_parameters(prm);
  this->add_parameter(prm, &mu, "Shear modulus", "10.0", Patterns::Double(0.0));
}

template <int dim, int spacedim, typename LAC>
void NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>::set_matrix_couplings(std::vector<std::string> &couplings) const
{
  couplings[0]="1,1;1,0";
  couplings[1]="1,0;0,1";
}

template <int dim, int spacedim, typename LAC>
void
NeoHookeanTwoFieldsInterface<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
    LinearOperator<LATrilinos::VectorType> &system_op,
    LinearOperator<LATrilinos::VectorType> &prec_op,
    LinearOperator<LATrilinos::VectorType> &) const
{

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector displacement(0);
  DoFTools::extract_constant_modes (this->get_dof_handler(),
                                    this->get_dof_handler().get_fe().component_mask(displacement),
                                    constant_modes);

  P_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.constant_modes = constant_modes;
  Amg_data.elliptic = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 0.02;

  P_preconditioner->initialize (matrices[1]->block(1,1));
  Amg_preconditioner->initialize (matrices[1]->block(0,0),Amg_data);


  // SYSTEM MATRIX:
  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[0]->block(0,0) );
  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[0]->block(0,1) );
  //  auto B =  transpose_operator(Bt);
  auto B     = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[0]->block(1,0) );
  auto ZeroP = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrices[0]->block(1,1) );

  auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[1]->block(1,1) );

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
  auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);
  auto Schur_inv = inverse_operator( Mp, solver_CG, *P_preconditioner);

  auto P00 = A_inv;
  auto P01 = null_operator(Bt);
  auto P10 = Schur_inv * B * A_inv;
  auto P11 = -1 * Schur_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<2, 2, LATrilinos::VectorType >({{
      {{ A, Bt }} ,
      {{ B, ZeroP }}
    }
  });


  //const auto S = linear_operator<VEC>(matrix);

  prec_op = block_operator<2, 2, LATrilinos::VectorType >({{
      {{ P00, P01 }} ,
      {{ P10, P11 }}
    }
  });
}

#endif
/*! @} */
