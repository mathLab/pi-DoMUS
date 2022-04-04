/*! \addtogroup equations
 *  @{
 */

/**
 *  Navier Stokes Equation using Chorin-Temam projection method
 */

#ifndef _pidomus_navier_stokes_h_
#define _pidomus_navier_stokes_h_

#include "pde_system_interface.h"

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal2lkit/sacado_tools.h>
#include <deal2lkit/parsed_preconditioner/amg.h>
#include <deal2lkit/parsed_preconditioner/jacobi.h>

////////////////////////////////////////////////////////////////////////////////
/// Navier Stokes interface:

template <int dim, int spacedim=dim, typename LAC=LATrilinos>
class NavierStokes
  :
  public PDESystemInterface<dim,spacedim,NavierStokes<dim,spacedim,LAC>, LAC>
{

public:
  ~NavierStokes () {};
  NavierStokes ();

  void declare_parameters (ParameterHandler &prm);

  template <typename EnergyType, typename ResidualType>
  void
  energies_and_residuals(
    const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    FEValuesCache<dim,spacedim> &scratch,
    std::vector<EnergyType> &energies,
    std::vector<std::vector<ResidualType>> &residuals,
    bool compute_only_system_terms) const;

  void
  compute_system_operators(
    const std::vector<shared_ptr<LATrilinos::BlockMatrix>>,
    LinearOperator<LATrilinos::VectorType> &,
    LinearOperator<LATrilinos::VectorType> &,
    LinearOperator<LATrilinos::VectorType> &) const;

  void
  set_matrix_couplings(std::vector<std::string> &couplings) const;

private:

  /**
   * AMG preconditioner for velocity-velocity matrix.
   */
  mutable ParsedJacobiPreconditioner AMG_u;

  /**
   * AMG preconditioner for velocity-velocity matrix.
   */
  mutable ParsedJacobiPreconditioner AMG_v;

  /**
   * AMG preconditioner for velocity-velocity matrix.
   */
  mutable ParsedAMGPreconditioner AMG_p;

  // PHYSICAL PARAMETERS:
  ////////////////////////////////////////////

  /**
   * Density
   */
  double rho;

  /**
   * Viscosity
   */
  double nu;

  double initial_delta_t;
  /**
  * Solver tolerance for CG
  */
  double CG_solver_tolerance;

  /**
   * Solver tolerance for GMRES
   */
  double GMRES_solver_tolerance;
};

template <int dim, int spacedim, typename LAC>
NavierStokes<dim,spacedim, LAC>::
NavierStokes()
  :
  PDESystemInterface<dim,spacedim,NavierStokes<dim,spacedim,LAC>, LAC>(
    "Navier Stokes Interface",
    dim+dim+1,
    1,
    "FESystem[FE_Q(2)^d-FE_Q(1)-FE_Q(2)^d]",
    (dim==3)?"v,v,v,p,u,u,u":"v,v,p,u,u",
    "1,0,1"),
  AMG_u("Prec for u", 1.4),
  AMG_v("Prec for v", 1.4),
  AMG_p("AMG for p")
{
  this->init();
}

template <int dim, int spacedim, typename LAC>
void NavierStokes<dim,spacedim,LAC>::
set_matrix_couplings(std::vector<std::string> &couplings) const
{
  couplings[0] = "1,1,1;1,1,1;1,1,1";
}

template <int dim, int spacedim, typename LAC>
void NavierStokes<dim,spacedim,LAC>::
declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, NavierStokes<dim,spacedim,LAC>,LAC>::
  declare_parameters(prm);
  this->add_parameter(prm, &rho,
                      "rho [kg m^3]", "1.0",
                      Patterns::Double(0.0),
                      "Density");
  this->add_parameter(prm, &nu,
                      "nu [Pa s]", "1.0",
                      Patterns::Double(0.0),
                      "Viscosity");
  this->add_parameter(prm, &initial_delta_t,
                      "initial delta t", "1e-1",
                      Patterns::Double(0.0),
                      "Initial Delta t");
  this->add_parameter(prm, &CG_solver_tolerance,
                      "CG Solver tolerance", "1e-8",
                      Patterns::Double(0.0));
  this->add_parameter(prm, &GMRES_solver_tolerance,
                      "GMRES Solver tolerance", "1e-8",
                      Patterns::Double(0.0));
}

template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
NavierStokes<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &residual,
                       bool compute_only_system_terms) const
{
  const FEValuesExtractors::Vector aux_velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);
  const FEValuesExtractors::Vector velocity(dim+1);

  ResidualType et = 0;
  double dummy = 0.0;

  this->reinit (et, cell, fe_cache);

  // Velocity:
  auto &us = fe_cache.get_values("solution", "u", velocity, et);
  auto &ues = fe_cache.get_values("explicit_solution", "ue", velocity, dummy);
  auto &grad_ues = fe_cache.get_gradients("explicit_solution", "grad_u", velocity, dummy);
  auto &div_us = fe_cache.get_divergences("solution", "div_u", velocity, et);
  auto &sym_grad_ues = fe_cache.get_symmetric_gradients("explicit_solution", "sym_grad_ue", velocity, dummy);

  auto &vs = fe_cache.get_values("solution", "v", aux_velocity, et);
  auto &ves = fe_cache.get_values("explicit_solution", "ve", aux_velocity, dummy);
  auto &sym_grad_vs = fe_cache.get_symmetric_gradients("solution", "sym_grad_v", aux_velocity, et);
  auto &grad_vs = fe_cache.get_gradients("solution", "grad_v", aux_velocity, et);
  auto &div_vs = fe_cache.get_divergences("solution", "div_v", aux_velocity, et);

  // Pressure:
  auto &ps = fe_cache.get_values("solution", "p", pressure, et);
  auto &pes = fe_cache.get_values("explicit_solution", "pe", pressure, dummy);
  auto &grad_ps = fe_cache.get_gradients("solution", "grad_p", pressure,et);
  auto &grad_pes = fe_cache.get_gradients("explicit_solution", "grad_pe", pressure,dummy);


  const unsigned int n_quad_points = us.size();
  auto &JxW = fe_cache.get_JxW_values();
  const auto delta_t = this->get_timestep() != this->get_timestep() ? initial_delta_t : this->get_timestep() ;
  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int quad=0; quad<n_quad_points; ++quad)
    {
      // // Delta Pressure:
      // const ResidualType &dp = dps[quad];
      // const Tensor<1, dim, ResidualType> &grad_dp = grad_dps[quad];

      // Pressure:
      const ResidualType &p = ps[quad];
      const double &pe = pes[quad];
      const Tensor<1, dim, ResidualType> &grad_p = grad_ps[quad];
      const Tensor<1, dim, double> &grad_pe = grad_pes[quad];

      // Velocity:
      const Tensor<1, dim, ResidualType> &u  = us[quad];
      const Tensor<1, dim, double> &ue = ues[quad];
      const Tensor<2, dim, double> &grad_ue = grad_ues[quad];
      const ResidualType &div_u = div_us[quad];

      const Tensor<1, dim, ResidualType> &v  = vs[quad];
      const Tensor<1, dim, double> vd  = SacadoTools::to_double(v);
      const Tensor<2, dim, ResidualType> &grad_v = grad_vs[quad];
      const Tensor<2, dim, ResidualType> &sym_grad_v = sym_grad_vs[quad];
      const Tensor<2, dim, double> &sym_grad_ue = sym_grad_ues[quad];
      const Tensor<1, dim, double> &ve = ves[quad];
      const ResidualType &div_v = div_vs[quad];

      for (unsigned int i=0; i<residual[0].size(); ++i)
        {
          // Velocity:
          auto v_test = fev[ aux_velocity ].value(i,quad);
          auto sym_grad_v_test = fev[ aux_velocity ].symmetric_gradient(i,quad);


          auto u_test = fev[ velocity ].value(i,quad);
          auto div_u_test = fev[ velocity ].divergence(i,quad);
          auto sym_grad_ue_test = fev[ velocity ].symmetric_gradient(i,quad);

          // Pressure:
          auto p_test = fev[ pressure ].value(i,quad);
          auto grad_p_test = fev[ pressure ].gradient(i,quad);

          ResidualType res = 0.0;

          res += ((v-ue)/delta_t)*v_test;
          res +=  (ue * grad_ue )*v_test;
          res += nu * scalar_product( sym_grad_ue,
                                      sym_grad_ue_test);

          /*           res +=  (ue * grad_v )*v_test; */
          /*           res += nu * scalar_product( sym_grad_v, */
          /*                                       sym_grad_v_test); */


          res +=  grad_p*grad_p_test + (rho/delta_t)*(div_v * p_test);

// Updating of the solution:
          res += (u - v + (delta_t/rho) *grad_p)*u_test;

          residual[0][i] += res * JxW[quad];
        }
    }
  (void)compute_only_system_terms;
}


template <int dim, int spacedim, typename LAC>
void
NavierStokes<dim,spacedim,LAC>::compute_system_operators(
  const std::vector<shared_ptr<LATrilinos::BlockMatrix>> matrices,
  LinearOperator<LATrilinos::VectorType> &system_op,
  LinearOperator<LATrilinos::VectorType> &prec_op,
  LinearOperator<LATrilinos::VectorType> &prec_op_finer) const
{
  const unsigned int num_blocks = 3;
  typedef LATrilinos::VectorType::BlockType  BVEC;
  typedef LATrilinos::VectorType             VEC;

  static ReductionControl solver_control(matrices[0]->m(), CG_solver_tolerance);
  static SolverCG<BVEC> solver_CG(solver_control);

  const DoFHandler<dim,spacedim> &dh = this->get_dof_handler();
  const ParsedFiniteElement<dim,spacedim> fe = this->pfe;
  AMG_v.initialize_preconditioner( matrices[0]->block(0,0)); //, fe, dh);
  AMG_p.initialize_preconditioner<dim, spacedim>( matrices[0]->block(1,1), fe, dh);
  AMG_u.initialize_preconditioner( matrices[0]->block(2,2)); //, fe, dh);

  ////////////////////////////////////////////////////////////////////////////
  // SYSTEM MATRIX:
  ////////////////////////////////////////////////////////////////////////////
  std::array<std::array<LinearOperator< BVEC >, num_blocks>, num_blocks> S;
  for (unsigned int i = 0; i<num_blocks; ++i)
    for (unsigned int j = 0; j<num_blocks; ++j)
      S[i][j] = linear_operator< BVEC >(matrices[0]->block(i,j) );
  system_op = BlockLinearOperator< VEC >(S);


  ////////////////////////////////////////////////////////////////////////////
  // PRECONDITIONER MATRIX:
  ////////////////////////////////////////////////////////////////////////////
  auto Av = linear_operator<BVEC>(matrices[0]->block(0,0));
  auto Av_inv  =  inverse_operator(Av, solver_CG, AMG_v);

  auto Ap = linear_operator<BVEC>(matrices[0]->block(1,1));
  auto Ap_inv  =  inverse_operator(Ap, solver_CG, AMG_p);

  auto Au = linear_operator<BVEC>(matrices[0]->block(2,2));
  auto Au_inv  =  inverse_operator(Au, solver_CG, AMG_u);

  // Preconditioner
  //////////////////////////////

  BlockLinearOperator<VEC> diag_inv
  = block_diagonal_operator<num_blocks, VEC>(
  {
    {
      Av_inv, Ap_inv, Au_inv
    }
  }
  );
  prec_op = diag_inv;

  /*   prec_op = block_forward_substitution( */
  /*               BlockLinearOperator< VEC >(S), */
  /*               diag_inv); */
}

#endif

/*! @} */
