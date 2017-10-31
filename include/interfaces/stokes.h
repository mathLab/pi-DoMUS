#ifndef __pidomus__stokes_h_
#define __pidomus__stokes_h_

#include "pde_system_interface.h"
#include <deal2lkit/parsed_function.h>

#include <deal.II/lac/solver_cg.h>
#include <deal2lkit/sacado_tools.h>
#include <deal2lkit/parsed_preconditioner/amg.h>
#include <deal2lkit/parsed_preconditioner/jacobi.h>

using namespace SacadoTools;

template <int dim, int spacedim, typename LAC>
class StokesInterface : public PDESystemInterface<dim,spacedim, StokesInterface<dim,spacedim,LAC>, LAC>
{
public:

  /* specific and useful functions for this very problem */
  StokesInterface();

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


  mutable ParsedAMGPreconditioner AMG_A;
  mutable ParsedJacobiPreconditioner jac_Mp;

};

template <int dim, int spacedim, typename LAC>
StokesInterface<dim,spacedim,LAC>::StokesInterface() :
  PDESystemInterface<dim,spacedim,StokesInterface<dim,spacedim,LAC>, LAC >("Stokes parameters",
      dim+1,2,
      "FESystem[FE_Q(2)^d-FE_Q(1)]",
      (spacedim==3?"u,u,u,p":"u,u,p"),
      "1,0"),
  AMG_A("Amg preconditioner for velocity"),
  jac_Mp("Jacobi preconditioner for pressure")
{}


template <int dim, int spacedim,typename LAC>
template<typename EnergyType,typename ResidualType>
void StokesInterface<dim,spacedim,LAC>::energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    FEValuesCache<dim,spacedim> &fe_cache,
    std::vector<EnergyType> &,
    std::vector<std::vector<ResidualType> > &local_residuals,
    bool compute_only_system_terms) const
{
  ResidualType alpha = 0;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);

  auto &grad_us = fe_cache.get_gradients("solution", "u", displacement, alpha);
//  auto &us = fe_cache.get_values("solution", "u", displacement, alpha);
  auto &div_us = fe_cache.get_divergences("solution", "u", displacement, alpha);

  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", displacement, alpha);

  const FEValuesExtractors::Scalar pressure(dim);
  auto &ps = fe_cache.get_values("solution","p", pressure, alpha);

  auto &JxW = fe_cache.get_JxW_values();
  auto &fev = fe_cache.get_current_fe_values();

  const unsigned int n_q_points = ps.size();

  for (unsigned int q=0; q<n_q_points; ++q)
    {

      const ResidualType &p = ps[q];
      const Tensor<2,dim,ResidualType> &grad_u = grad_us[q];
//      const Tensor<1,dim,ResidualType> &u = us[q];
      const Tensor<1,dim,ResidualType> &u_dot = us_dot[q];
      const ResidualType &div_u = div_us[q];

      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
          // v is test for u (displacement)
          // q_test is test for p (pressure)
          auto v = fev[displacement].value(i,q);
          auto grad_v = fev[displacement].gradient(i,q);
          auto div_v = fev[displacement].divergence(i,q);
          auto q_test = fev[pressure].value(i,q);

          local_residuals[0][i] += (
                                     u_dot*v
                                     + mu*scalar_product(grad_u,grad_v)
                                     - p*div_v
                                     + div_u*q_test
                                   )*JxW[q];
          // needed by preconditioner --- this can be improved for time-dep problems
          if (!compute_only_system_terms)
            local_residuals[1][i] += (p*q_test)*JxW[q];

        }
    }
}


template <int dim, int spacedim, typename LAC>
void StokesInterface<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, StokesInterface<dim,spacedim,LAC>,LAC >::declare_parameters(prm);
  this->add_parameter(prm, &mu, "Viscosity", "1.0", Patterns::Double(0.0));
}

template <int dim, int spacedim, typename LAC>
void StokesInterface<dim,spacedim,LAC>::set_matrix_couplings(std::vector<std::string> &couplings) const
{
  couplings[0]="1,1;1,0";
  couplings[1]="0,0;0,1";
}

template <int dim, int spacedim, typename LAC>
void
StokesInterface<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
                                                            LinearOperator<LATrilinos::VectorType> &system_op,
                                                            LinearOperator<LATrilinos::VectorType> &prec_op,
                                                            LinearOperator<LATrilinos::VectorType> &) const
{

  const DoFHandler<dim,spacedim> &dh = this->get_dof_handler();
  const ParsedFiniteElement<dim,spacedim> &fe = this->pfe;

  AMG_A.initialize_preconditioner<dim, spacedim>( matrices[0]->block(0,0), fe, dh);

  jac_Mp.initialize_preconditioner<>(matrices[1]->block(1,1));


  // SYSTEM MATRIX:
  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[0]->block(0,0) );
  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[0]->block(0,1) );
  //  auto B =  transpose_operator(Bt);
  auto B     = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[0]->block(1,0) );
  auto ZeroP = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrices[0]->block(1,1) );

  auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[1]->block(1,1) );

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
  auto A_inv = linear_operator<TrilinosWrappers::MPI::Vector>(matrices[0]->block(0,0), AMG_A);
  auto Schur_inv = inverse_operator( Mp, solver_CG, jac_Mp);

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
