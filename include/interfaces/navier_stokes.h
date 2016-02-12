/*! \addtogroup equations
 *  @{
 */

/**
 *  This interface solves a Navier Stokes Equation:
 *  \f[
 *     \begin{cases}
 *       \partial_t u + (u\cdot\nabla)u - \nu\textrm{div} \epsilon(u)
 *     + \frac{1}{\rho}\nabla p = f \\
 *       \textrm{div}u=0
 *     \end{cases}
 *  \f]
 *  where \f$ \epsilon(u) = \frac{\nabla u + [\nabla u]^t}{2}. \f$
 *
 *  Non time-depending Navier Stokes Equation:
 *  \f[
 *     \begin{cases}
 *       (u\cdot\nabla)u - \nu\textrm{div} \epsilon(u)
 *     + \frac{1}{\rho}\nabla p = f \\
 *       \textrm{div}u=0
 *     \end{cases}
 *  \f]
 *  can be recoverd setting @p dynamic = false
 *
 *  Dynamic Stokes Equation:
 *  \f[
 *     \begin{cases}
 *       \partial_t u - \nu\textrm{div} \epsilon(u)
 *     + \frac{1}{\rho}\nabla p = f \\
 *       \textrm{div}u=0
 *     \end{cases}
 *  \f]
 *  can be recoverd setting @p convection = false
 *
 *  Stokes Equation:
 *  \f[
 *     \begin{cases}
 *       - \nu\textrm{div} \epsilon(u)
 *     + \frac{1}{\rho}\nabla p = f \\
 *       \textrm{div}u=0
 *     \end{cases}
 *  \f]
 *  can be recoverd setting @p dynamic = false and @p convection = false
 *
 * In the code we adopt the following notations:
 * - Mp := block resulting from \f$ ( \partial_t p, q ) \f$
 * - Ap := block resulting from \f$ \nu ( \nabla p,\nabla q ) \f$
 * - Np := block resulting from \f$ ( u \cdot \nabla p, q) \f$
 *
 * where:
 * - p = pressure
 * - q = test function for the pressure
 * - u = velocity
 * - v = test function for the velocity
 *
 * Notes on preconditioners:
 * - default: This preconditioner uses the mass matrix of pressure block as
 * inverse for the Schur block.
 * This is a preconditioner suitable for problems wher the viscosity is
 * higher than the density. \f[ S^-1 = \nu M_p \f]
 * - identity: Identity matrix preconditioner
 * - low-nu: This preconditioner uses the stifness matrix of pressure block
 * as inverse for the Schur block. \f[ S^-1 = \rho \frac{1}{\Delta t} A_p \f]
 * - cah-cha:  Preconditioner suggested by J. Cahouet and J.-P. Chabard.
 *  \f[ S^-1 =  \nu M_p  + \rho \frac{1}{\Delta t} A_p. \f]
 *
 */

#ifndef _pidomus_navier_stokes_h_
#define _pidomus_navier_stokes_h_

#include "pde_system_interface.h"

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

template <int dim, int spacedim=dim, typename LAC=LATrilinos>
class NavierStokes
  :
  public PDESystemInterface<dim,spacedim,NavierStokes<dim,spacedim,LAC>, LAC>
{

public:
  ~NavierStokes () {};
  NavierStokes (bool dynamic, bool convection);

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

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
    LinearOperator<LATrilinos::VectorType> &) const;

  void
  set_matrix_couplings(std::vector<std::string> &couplings) const;

private:
  /**
   * Enable dynamic term: \f$ \partial_t u\f$.
   */
  bool dynamic;

  /**
   * Enable convection term: \f$ (\nabla u)u \f$.
   * This term introduce the non-linearity of Navier Stokes Equations.
   */
  bool convection;

  /**
   * Hot to handle \f$ (\nabla u)u \f$.
   */
  std::string non_linear_term;

  /**
   * Hot to handle \f$ (\nabla u)u \f$.
   */
  bool linearize_in_time;

  /**
  * Name of the preconditioner:
  */
  std::string prec_name;

  /**
   * Density
   */
  double rho;

  /**
   * Viscosity
   */
  double nu;

  /**
   * div-grad stabilization parameter
   */
  double gamma;

  /**
   * p-q stabilization parameter
   */
  double gamma_p;

  /**
   * Compute Mp
   */
  bool compute_Mp;

  /**
   * Compute Ap
   */
  bool compute_Ap;

  /**
   * Invert Mp using inverse_operator
   */
  bool invert_Mp;

  /**
   * Invert Ap using inverse_operator
   */
  bool invert_Ap;

  /**
  * Solver tolerance for CG
  */
  double CG_solver_tolerance;

  /**
   * Solver tolerance for GMRES
   */
  double GMRES_solver_tolerance;

  /**
   * AMG smoother sweeps:
   */
  int amg_A_smoother_sweeps;

  /**
   * AMG aggregation threshold:
   */
  double amg_A_aggregation_threshold;

  /**
   * AMG elliptic:
   */
  bool amg_A_elliptic;

  /**
   * AMG high order elements:
   */
  bool amg_A_higher_order_elements;

  /**
   * AMG smoother sweeps:
   */
  int amg_Ap_smoother_sweeps;

  /**
   * AMG aggregation threshold:
   */
  double amg_Ap_aggregation_threshold;

  /**
   * AMG elliptic:
   */
  bool amg_Ap_elliptic;

  mutable shared_ptr<TrilinosWrappers::PreconditionAMG> amg_A;
  mutable shared_ptr<TrilinosWrappers::PreconditionAMG> amg_Ap;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> jacobi_Mp;
};

template <int dim, int spacedim, typename LAC>
NavierStokes<dim,spacedim, LAC>::
NavierStokes(bool dynamic, bool convection)
  :
  PDESystemInterface<dim,spacedim,NavierStokes<dim,spacedim,LAC>, LAC>(
    "Navier Stokes Interface",
    dim+1,
    3,
    "FESystem[FE_Q(2)^d-FE_Q(1)]",
    "u,u,p",
    "1,0"),
  dynamic(dynamic),
  convection(convection),
  compute_Mp(false),
  compute_Ap(false)
{
  this->init();
}

template <int dim, int spacedim, typename LAC>
void NavierStokes<dim,spacedim,LAC>::
declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, NavierStokes<dim,spacedim,LAC>,LAC>::
  declare_parameters(prm);
  this->add_parameter(prm, &prec_name,  "Preconditioner","default",
                      Patterns::Selection("default|identity|low-nu|cah-cha"),
                      "Available preconditioners: \n"
                      " - default  -> S^-1 = nu * Mp^-1 \n"
                      " - identity -> S^-1 = identity \n"
                      " - low-nu   -> S^-1 = rho * alpha * Ap^-1 \n"
                      " - cah-cha  -> S^-1 = nu * Mp^-1 + rho * alpha * Ap^-1 ");
  this->add_parameter(prm, &dynamic,
                      "Enable dynamic term (\\partial_t u)", "true",
                      Patterns::Bool(),
                      "Enable the dynamic term of the equation.");
  this->add_parameter(prm, &convection,
                      "Enable convection term ((\\nabla u)u)", "true",
                      Patterns::Bool(),
                      "Enable the convection term of the equation. Set it false if you want to solve Stokes Equation.");
  this->add_parameter(prm, &non_linear_term, "Non linear term","linear",
                      Patterns::Selection("fully_non_linear|linear|RHS"),
                      "Available options: \n"
                      " fully_non_linear\n"
                      "u_linear\n"
                      "RHS\n");
  this->add_parameter(prm, &linearize_in_time,
                      "Linearize using time", "true",
                      Patterns::Bool(),
                      "If true use the solution of the previous time step\n"
                      "to linearize the non-linear term, otherwise use the\n" "solution of the previous step (of an iterative methos).");
  this->add_parameter(prm, &gamma,
                      "div-grad stabilization parameter", "0.0",
                      Patterns::Double(0.0),
                      "");
  this->add_parameter(prm, &gamma_p,
                      "p-q stabilization parameter", "0.0",
                      Patterns::Double(0.0),
                      "");
  this->add_parameter(prm, &rho,
                      "rho [kg m^3]", "1.0",
                      Patterns::Double(0.0),
                      "Density");
  this->add_parameter(prm, &nu,
                      "nu [Pa s]", "1.0",
                      Patterns::Double(0.0),
                      "Viscosity");
  this->add_parameter(prm, &invert_Ap,
                      "Invert Ap using inverse_operator", "true", Patterns::Bool());
  this->add_parameter(prm, &invert_Mp,
                      "Invert Mp using inverse_operator", "true", Patterns::Bool());
  this->add_parameter(prm, &CG_solver_tolerance,
                      "CG Solver tolerance", "1e-8",
                      Patterns::Double(0.0));
  this->add_parameter(prm, &GMRES_solver_tolerance,
                      "GMRES Solver tolerance", "1e-8",
                      Patterns::Double(0.0));
  this->add_parameter(prm, &amg_A_smoother_sweeps,
                      "A - Amg Smoother Sweeps","2", Patterns::Integer(0));
  this->add_parameter(prm, &amg_A_aggregation_threshold,
                      "A - Amg Aggregation Threshold", "0.02", Patterns::Double(0.0));
  this->add_parameter(prm, &amg_A_elliptic,
                      "A - Amg Elliptic", "true", Patterns::Bool());
  this->add_parameter(prm, &amg_A_higher_order_elements,
                      "A - Amg High Order Elements", "true", Patterns::Bool());
  this->add_parameter(prm, &amg_Ap_smoother_sweeps,
                      "Ap - Amg Smoother Sweeps","2", Patterns::Integer(0));
  this->add_parameter(prm, &amg_Ap_aggregation_threshold,
                      "Ap - Amg Aggregation Threshold", "0.02", Patterns::Double(0.0));
  this->add_parameter(prm, &amg_Ap_elliptic,
                      "Ap - Amg Elliptic", "true", Patterns::Bool());
}

template <int dim, int spacedim, typename LAC>
void NavierStokes<dim,spacedim,LAC>::
parse_parameters_call_back ()
{
  if (prec_name == "default")
    compute_Mp = true;
  else if (prec_name == "low-nu")
    compute_Ap = true;
  else if (prec_name == "cah-cha")
    {
      compute_Mp = true;
      compute_Ap = true;
    }

  // p-q stabilization term:
  if (gamma_p!=0.0)
    compute_Mp = true;
}

template <int dim, int spacedim, typename LAC>
void NavierStokes<dim,spacedim,LAC>::
set_matrix_couplings(std::vector<std::string> &couplings) const
{
  couplings[0] = "1,1;1,0"; // Direct solver uses block(1,0)
  couplings[1] = "0,0;0,1";
  couplings[2] = "0,0;0,1";
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
  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);

  ResidualType et = this->alpha;
  double dummy = 0.0;
  // dummy number to define the type of variables
  this->reinit (et, cell, fe_cache);

  // Velocity:
  auto &us = fe_cache.get_values("solution", "u", velocity, et);
  auto &div_us = fe_cache.get_divergences("solution", "div_u", velocity,et);
  auto &grad_us = fe_cache.get_gradients("solution", "grad_u", velocity,et);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients("solution", "sym_grad_u", velocity,et);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", velocity, et);

  // Velocity:
  auto &ps = fe_cache.get_values("solution", "p", pressure, et);
  auto &grad_ps = fe_cache.get_gradients("solution", "grad_p", pressure,et);

  // Previous time step solution:
  auto &u_olds = fe_cache.get_values("explicit_solution", "u", velocity, dummy);
  auto &grad_u_olds = fe_cache.get_gradients("explicit_solution", "grad_u", velocity, dummy);

  // Previous Jacobian step solution:
  fe_cache.cache_local_solution_vector("prev_solution", *this->solution, dummy);
  auto &u_prevs = fe_cache.get_values("prev_solution", "u", velocity, dummy);
  auto &grad_u_prevs = fe_cache.get_gradients("prev_solution", "grad_u", velocity, dummy);

  const unsigned int n_quad_points = us.size();
  auto &JxW = fe_cache.get_JxW_values();

  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int quad=0; quad<n_quad_points; ++quad)
    {
      // Pressure:
      const ResidualType &p = ps[quad];
      const Tensor<1, dim, ResidualType> &grad_p = grad_ps[quad];

      // Velocity:
      const Tensor<1, dim, ResidualType> &u = us[quad];
      const Tensor<1, dim, ResidualType> &u_dot = us_dot[quad];
      const Tensor<2, dim, ResidualType> &grad_u = grad_us[quad];
      const Tensor<2, dim, ResidualType> &sym_grad_u = sym_grad_us[quad];
      const ResidualType &div_u = div_us[quad];

      // Previous time step solution:
      const Tensor<1, dim, ResidualType> &u_old = u_olds[quad];
      const Tensor<2, dim, ResidualType> &grad_u_old = grad_u_olds[quad];

      // Previous Jacobian step solution:
      const Tensor<1, dim, ResidualType> &u_prev = u_prevs[quad];
      const Tensor<2, dim, ResidualType> &grad_u_prev = grad_u_prevs[quad];

      for (unsigned int i=0; i<residual[0].size(); ++i)
        {
          // Velocity:
          auto v = fev[velocity ].value(i,quad);
          auto div_v = fev[velocity ].divergence(i,quad);
          auto sym_grad_v = fev[ velocity ].symmetric_gradient(i,quad);

          // Pressure:
          auto q = fev[ pressure ].value(i,quad);
          auto grad_q = fev[ pressure ].gradient(i,quad);

          // Non-linear term:
          Tensor<1, dim, ResidualType> nl_u;
          ResidualType res = 0.0;

          // Time derivative:
          if (dynamic)
            res += rho * u_dot * v;

          // Convection:
          if (convection)
            {
              Tensor<2, dim, ResidualType> gradoldu;
              Tensor<1, dim, ResidualType> oldu;

              if (linearize_in_time)
                {
                  gradoldu=grad_u_old;
                  oldu=u_old;
                }
              else
                {
                  gradoldu=grad_u_prev;
                  oldu=u_prev;
                }

              if (non_linear_term=="fully_non_linear")
                nl_u = grad_u * u;
              else if (non_linear_term=="linear")
                nl_u = grad_u * oldu;
              else if (non_linear_term=="RHS")
                nl_u = gradoldu * oldu;

              res += rho * scalar_product(nl_u,v);
            }
          // grad-div stabilization term:
          if (gamma!=0.0)
            res += gamma * div_u * div_v;

          // Diffusion term:
          res += nu * scalar_product(sym_grad_u, sym_grad_v);

          // Pressure term:
          res -= p * div_v;

          // Incompressible constraint:
          res -= div_u * q;

          residual[0][i] += res * JxW[q];

          // Mp preconditioner:
          if (!compute_only_system_terms && compute_Mp)
            residual[1][i] += p * q * JxW[q];

          // Ap preconditioner:
          if (!compute_only_system_terms && compute_Ap)
            residual[2][i] += grad_p * grad_q * JxW[q];
        }
    }
  (void)compute_only_system_terms;
}


template <int dim, int spacedim, typename LAC>
void
NavierStokes<dim,spacedim,LAC>::compute_system_operators(
  const std::vector<shared_ptr<LATrilinos::BlockMatrix>> matrices,
  LinearOperator<LATrilinos::VectorType> &system_op,
  LinearOperator<LATrilinos::VectorType> &prec_op) const
{
  auto aplha = this->alpha;

  typedef LATrilinos::VectorType::BlockType  BVEC;
  typedef LATrilinos::VectorType             VEC;

  static ReductionControl solver_control_cg(matrices[0]->m(), CG_solver_tolerance);
  static SolverCG<BVEC> solver_CG(solver_control_cg);

  static ReductionControl solver_control_gmres(matrices[0]->m(), GMRES_solver_tolerance);
  static SolverGMRES<BVEC> solver_GMRES(solver_control_gmres);


  // SYSTEM MATRIX:
  auto A  = linear_operator<BVEC>( matrices[0]->block(0,0) );
  auto Bt = linear_operator<BVEC>( matrices[0]->block(0,1) );
  auto B  = transpose_operator<BVEC>( Bt );
  auto C  = B*Bt;
  auto ZeroP = null_operator(C);

  if (gamma_p!=0.0)
    C = linear_operator<BVEC>( matrices[1]->block(1,1) );
  else
    C = ZeroP;

  // ASSEMBLE THE PROBLEM:
  system_op = block_operator<2, 2, VEC>({{
      {{ A, Bt }} ,
      {{ B, C }}
    }
  });

  std::vector<std::vector<bool>> constant_modes;
  FEValuesExtractors::Vector velocity_components(0);
  DoFTools::extract_constant_modes (*this->dof_handler, this->dof_handler->get_fe().component_mask(velocity_components),
                                    constant_modes);
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_A_data;
  amg_A_data.constant_modes = constant_modes;
  amg_A_data.elliptic = amg_A_elliptic;
  amg_A_data.higher_order_elements = amg_A_higher_order_elements;
  amg_A_data.smoother_sweeps = amg_A_smoother_sweeps;
  amg_A_data.aggregation_threshold = amg_A_aggregation_threshold;

  std::vector<std::vector<bool> > constant_modes_p;
  FEValuesExtractors::Scalar pressure_components(dim);
  DoFTools::extract_constant_modes (*this->dof_handler, this->dof_handler->get_fe().component_mask(pressure_components),
                                    constant_modes_p);
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_Ap_data;
  amg_Ap_data.constant_modes = constant_modes_p;
  amg_Ap_data.elliptic = amg_Ap_elliptic;
  amg_Ap_data.smoother_sweeps = amg_Ap_smoother_sweeps;
  amg_Ap_data.aggregation_threshold = amg_Ap_aggregation_threshold;


  amg_A.reset (new TrilinosWrappers::PreconditionAMG());
  amg_A->initialize (matrices[0]->block(0,0), amg_A_data);
  auto A_inv = inverse_operator( A, solver_GMRES, *amg_A);

  LinearOperator<BVEC> Schur_inv;
  LinearOperator<BVEC> Ap, Ap_inv, Mp, Mp_inv;

  if (compute_Mp)
    {
      Mp = linear_operator<BVEC>( matrices[1]->block(1,1) );
      jacobi_Mp.reset (new TrilinosWrappers::PreconditionJacobi());
      jacobi_Mp->initialize (matrices[1]->block(1,1), 1.4);
      Mp_inv = inverse_operator( Mp, solver_CG, *jacobi_Mp);
    }
  if (compute_Ap)
    {
      Mp = linear_operator<BVEC>( matrices[2]->block(1,1) );
      amg_Ap.reset (new TrilinosWrappers::PreconditionAMG());
      amg_Ap->initialize (matrices[2]->block(1,1),  amg_Ap_data);
      if (invert_Ap)
        {
          Ap_inv  = inverse_operator( Ap, solver_CG, *amg_Ap);
        }
      else
        {
          Ap_inv = linear_operator<BVEC>(matrices[2]->block(1,1), *amg_Ap);
        }
    }

  LinearOperator<BVEC> P00,P01,P10,P11;

  if (prec_name=="default")
    Schur_inv = (gamma + nu) * Mp_inv;
  else if (prec_name=="low-nu")
    Schur_inv = aplha * rho * Ap_inv;
  else if (prec_name=="identity")
    Schur_inv = identity_operator((C).reinit_range_vector);
  else if (prec_name=="cha-cha")
    Schur_inv = (gamma + nu) * Mp_inv + aplha * rho * Ap_inv;

  P00 = A_inv;
  P01 = A_inv * Bt * Schur_inv;
  P10 = null_operator(B);
  P11 = -1 * Schur_inv;

  prec_op = block_operator<2, 2, VEC>({{
      {{ P00, P01 }} ,
      {{ P10, P11 }}
    }
  });
}

#endif

/*! @} */
