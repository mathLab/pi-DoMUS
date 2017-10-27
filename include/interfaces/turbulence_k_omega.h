/*! \addtogroup equations
 *  @{
 */

/**
 *  see Wilcox, D.C. (1988), "Re-assessment of the scale-determining
 *  equation for advanced turbulence models", AIAA Journal, vol. 26,
 *  no. 11, pp. 1299-1310.
 */

#ifndef _pidomus_turbulence_k_omega_h_
#define _pidomus_turbulence_k_omega_h_

#include "pde_system_interface.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal2lkit/sacado_tools.h>
#include <deal2lkit/parsed_preconditioner/amg.h>
#include <deal2lkit/parsed_preconditioner/jacobi.h>

template <int dim, int spacedim=dim, typename LAC=LATrilinos>
class KOmega
  :
  public PDESystemInterface<dim,spacedim,KOmega<dim,spacedim,LAC>, LAC>
{

public:
  ~KOmega () {}
  KOmega ();

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
    const std::vector<shared_ptr<LATrilinos::BlockMatrix>> &,
    LinearOperator<LATrilinos::VectorType> &,
    LinearOperator<LATrilinos::VectorType> &,
    LinearOperator<LATrilinos::VectorType> &) const;

  void
  set_matrix_couplings(std::vector<std::string> &couplings) const;

private:
  /**
   * Determine to handle \f$ (\nabla u)u \f$.
   */
  std::string non_linear_term;

  /**
   * Determine to approximate the previous step in \f$ (\nabla u)u
   * \f$.
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
   * Turbulent Viscosity
   */
  mutable double nu_T;

  /**
   * alpha - turbulent variable of k-omega model
   */
  double alpha;

  /**
   * beta - turbulent variable of k-omega model
   */
  double beta;

  /**
   * beta_* - turbulent variable of k-omega model
   */
  double beta_star;

  /**
   * sigma - turbulent variable of k-omega model
   */
  double sigma;

  /**
   * sigma_* - turbulent variable of k-omega model
   */
  double sigma_star;

  // STABILIZATION:
  ////////////////////////////////////////////
  /**
   * div-grad stabilization parameter
   */
  double gamma;

  /**
   * SUPG stabilization term
   */
  double SUPG_alpha;

  /**
   * p-q stabilization parameter
   */
  double gamma_p;

  /**
  * Solver tolerance for CG
  */
  double CG_solver_tolerance;

  /**
   * Solver tolerance for GMRES
   */
  double GMRES_solver_tolerance;

  /**
   * AMG preconditioner for velocity-velocity matrix.
   */
  mutable ParsedAMGPreconditioner amg_A;

  /**
   * AMG preconditioner for the pressure stifness matrix.
   */
  mutable ParsedAMGPreconditioner amg_Ap;

  /**
   * AMG preconditioner for the k-k block.
   */
  mutable ParsedAMGPreconditioner amg_k;

  /**
   * AMG preconditioner for the omega-omega block.
   */
  mutable ParsedAMGPreconditioner amg_w;

  /**
   * Jacobi preconditioner for the pressure mass matrix.
   */
  mutable ParsedJacobiPreconditioner jacobi_Mp;
};

template <int dim, int spacedim, typename LAC>
KOmega<dim,spacedim, LAC>::
KOmega()
  :
  PDESystemInterface<dim,spacedim,KOmega<dim,spacedim,LAC>, LAC>(
    "k-omega",
    dim+3,
    3,
    "FESystem[FE_Q(2)^d-FE_Q(1)-FE_Q(1)-FE_Q(1)]",
    (dim==3) ? "u,u,u,p,k,w" : "u,u,p,k,w",
    "1,0,1,1"),
  amg_A("Amg for A"),
  amg_Ap("Amg for Ap"),
  amg_k("Amg for k"),
  amg_w("Amg for omega"),
  jacobi_Mp("Jacobi for Mp", 1.4)
{
  this->init();
}

template <int dim, int spacedim, typename LAC>
void KOmega<dim,spacedim,LAC>::
declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, KOmega<dim,spacedim,LAC>,LAC>::
  declare_parameters(prm);

  this->add_parameter(prm, &non_linear_term, "Non linear term","linear",
                      Patterns::Selection("fully_non_linear|linear|RHS"),
                      "Available options: \n"
                      " fully_non_linear\n"
                      " linear\n"
                      " RHS\n");
  this->add_parameter(prm, &gamma,
                      "div-grad stabilization parameter", "0.0",
                      Patterns::Double(0.0),
                      "");
  this->add_parameter(prm, &SUPG_alpha,
                      "SUPG alpha", "0.0",
                      Patterns::Double(0.0),
                      "Use SUPG alpha");
  this->add_parameter(prm, &rho,
                      "rho [kg m^3]", "1.0",
                      Patterns::Double(0.0),
                      "Density");
  this->add_parameter(prm, &nu,
                      "nu [Pa s]", "1.0",
                      Patterns::Double(0.0),
                      "Viscosity");
  this->add_parameter(prm, &alpha,
                      "alpha", "0.5555555556",
                      Patterns::Double(0.0),
                      "Alpha");
  this->add_parameter(prm, &beta,
                      "beta", "0.075",
                      Patterns::Double(0.0),
                      "Beta");
  this->add_parameter(prm, &beta_star,
                      "beta_star", "0.09",
                      Patterns::Double(0.0),
                      "Beta_*");
  this->add_parameter(prm, &sigma,
                      "sigma", ".5",
                      Patterns::Double(0.0),
                      "Sigma");
  this->add_parameter(prm, &sigma_star,
                      "sigma_star", ".5",
                      Patterns::Double(0.0),
                      "Sigma_*");
  this->add_parameter(prm, &CG_solver_tolerance,
                      "CG Solver tolerance", "1e-8",
                      Patterns::Double(0.0));
  this->add_parameter(prm, &GMRES_solver_tolerance,
                      "GMRES Solver tolerance", "1e-8",
                      Patterns::Double(0.0));
}

template <int dim, int spacedim, typename LAC>
void KOmega<dim,spacedim,LAC>::
parse_parameters_call_back ()
{}

template <int dim, int spacedim, typename LAC>
void KOmega<dim,spacedim,LAC>::
set_matrix_couplings(std::vector<std::string> &couplings) const
{
  //couplings[0] = "1,1,1,1;1,1,1,1;1,1,1,1"; // Direct solver uses block(1,0)
  //couplings[1] = "0,0,0,0;0,1,0,0;0,0,0,0";
  //couplings[2] = "0,0,0,0;0,1,0,0;0,0,0,0";
  couplings[0] = "1,1,1,1; 1,0,1,1; 1,1,1,1; 1,1,1,1";
  couplings[1] = "0,0,0,0; 0,1,0,0; 0,0,0,0; 0,0,0,0";
  couplings[2] = "0,0,0,0; 0,1,0,0; 0,0,0,0; 0,0,0,0"; // TODO: fix me!
}

template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
KOmega<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &residual,
                       bool compute_only_system_terms) const
{
  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);
  const FEValuesExtractors::Scalar kinetic_energy(dim+1);
  const FEValuesExtractors::Scalar turbulence_frequency(dim+2);

  double h = cell->diameter();

  ResidualType et = 0;
  double dummy = 0.0;
  // dummy number to define the type of variables
  this->reinit (et, cell, fe_cache);

  // Velocity:
  auto &u_ = fe_cache.get_values("solution", "u", velocity, et);
  auto &div_u_ = fe_cache.get_divergences("solution", "div_u", velocity, et);
  auto &grad_u_ = fe_cache.get_gradients("solution", "grad_u", velocity, et);
  auto &sym_grad_u_ = fe_cache.get_symmetric_gradients("solution", "sym_grad_u", velocity, et);
  auto &u_dot_ = fe_cache.get_values("solution_dot", "u_dot", velocity, et);

  // Velocity:
  auto &p_ = fe_cache.get_values("solution", "p", pressure, et);
  auto &grad_p_ = fe_cache.get_gradients("solution", "grad_p", pressure,et);

  // Kinetic Energy:
  auto &k_ = fe_cache.get_values("solution", "k", kinetic_energy, et);
  auto &grad_k_ = fe_cache.get_gradients("solution", "grad_k", kinetic_energy, et);
  auto &k_dot_ = fe_cache.get_values("solution_dot", "k_dot", kinetic_energy, et);

  // Turbulence Frequency:
  auto &w_ = fe_cache.get_values("solution", "w", turbulence_frequency, et);
  auto &grad_w_ = fe_cache.get_gradients("solution", "grad_w", turbulence_frequency, et);
  auto &w_dot_ = fe_cache.get_values("solution_dot", "w_dot", turbulence_frequency, et);

  // Previous time step solution:
  auto &ue_ = fe_cache.get_values("explicit_solution", "u", velocity, dummy);
  auto &we_ = fe_cache.get_values("explicit_solution", "w", turbulence_frequency, dummy);
  auto &ke_ = fe_cache.get_values("explicit_solution", "k", kinetic_energy, dummy);
  auto &grad_ke_ = fe_cache.get_gradients("explicit_solution", "grad_k", kinetic_energy, dummy);
  auto &grad_we_ = fe_cache.get_gradients("explicit_solution", "grad_w", turbulence_frequency, dummy);
  auto &grad_ue_ = fe_cache.get_gradients("explicit_solution", "grad_u", velocity, dummy);
  // auto &div_ue_ = fe_cache.get_divergences("explicit_solution", "div_u", velocity, dummy);
  // auto &sym_grad_ue_ = fe_cache.get_symmetric_gradients("explicit_solution", "sym_grad_u", velocity, dummy);

  const unsigned int n_quad_points = u_.size();
  auto &JxW = fe_cache.get_JxW_values();

  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int q=0; q<n_quad_points; ++q)
    {
      // Pressure:
      const ResidualType &p = p_[q];
      const Tensor<1, dim, ResidualType> &grad_p = grad_p_[q];

      // Velocity:
      const Tensor<1, dim, ResidualType> &u = u_[q];
      const Tensor<1, dim, ResidualType> &u_dot = u_dot_[q];
      const Tensor<2, dim, ResidualType> &grad_u = grad_u_[q];
      const Tensor<2, dim, ResidualType> &sym_grad_u = sym_grad_u_[q];
      const ResidualType &div_u = div_u_[q];

      // Turbulence_Frequency:
      const ResidualType &w = w_[q];
      const ResidualType &w_dot = w_dot_[q];
      const Tensor<1, dim, ResidualType> &grad_w = grad_w_[q];
      const Tensor<1, dim, double> &grad_we = grad_we_[q];

      // Kinetic Energy:
      const ResidualType &k = k_[q];
      const ResidualType &k_dot = k_dot_[q];
      const Tensor<1, dim, ResidualType> &grad_k = grad_k_[q];
      const Tensor<1, dim, double> &grad_ke = grad_ke_[q];

      // Previous time step solution:
      const Tensor<1, dim, double> &ue = ue_[q];
      const double &ke = ke_[q];
      const double &we = we_[q];
      const Tensor<2, dim, double> &grad_ue = grad_ue_[q];
      // const double &div_ue = div_ue_[q];
      // const Tensor<2, dim, double> &sym_grad_ue = sym_grad_ue_[q];

      for (unsigned int i=0; i<residual[0].size(); ++i)
        {
          // Velocity:
          auto u_test = fev[velocity].value(i,q);
          auto div_u_test = fev[velocity].divergence(i,q);
          auto grad_u_test = fev[velocity].gradient(i,q);
          auto sym_grad_u_test = fev[velocity].symmetric_gradient(i,q);

          // Pressure:
          auto p_test = fev[pressure].value(i,q);
          auto grad_p_test = fev[pressure].gradient(i,q);

          // Kappa:
          auto k_test = fev[kinetic_energy].value(i,q);
          auto grad_k_test = fev[kinetic_energy].gradient(i,q);

          // Turbulence_Frequency:
          auto w_test = fev[turbulence_frequency].value(i,q);
          auto grad_w_test = fev[turbulence_frequency].gradient(i,q);

          // Non-linear term:
          Tensor<1, dim, ResidualType> nl_u;
          ResidualType res = 0.0;

          // Generic Tensors:
          //////////////////////////////

          // Identity tensor:
          Tensor<2, dim, ResidualType> Id;
          for (unsigned int i = 0; i<dim; ++i)
            Id[i][i] = 1.0;

          // Turbulence:
          //////////////////////////////

          // Tensor<2, dim, ResidualType> S = sym_grad_ue;

          // Turbulent viscosity:
          nu_T = ke/we;

          // Reynold Stress Tensor
          Tensor<2, dim, ResidualType> tau = 2 * nu_T * sym_grad_u;
          for (unsigned int i = 0; i<dim; ++i)
            tau[i][i] -= 2 * nu_T *1./3. * div_u + 2./3. * ke;

          // Tensor<2, dim, ResidualType> tau = 2 * nu_T * ( S - (1./3. * div_u) * Id) - 2./3. * rho * k * Id; -> Not working

          // -> Navier Stokes:
          //////////////////////////////

          // Time derivative:
          res += rho * u_dot * u_test;

          // Convection:
          if (non_linear_term=="fully_non_linear")
            nl_u = grad_u * u;
          else if (non_linear_term=="linear")
            nl_u = grad_u * ue;
          else if (non_linear_term=="RHS")
            nl_u = grad_ue * ue;
          double norm = std::sqrt(SacadoTools::to_double(ue*ue));

          if (norm > 0 && SUPG_alpha > 0)
            res += rho * scalar_product( nl_u, u_test + SUPG_alpha * (h/norm) * grad_u_test  * ue);
          else
            res += rho * scalar_product( nl_u, u_test);


          // Turbulent term:
          res += rho * scalar_product(tau, grad_u_test);

          // grad-div stabilization term:
          if (gamma!=0.0)
            res += gamma * div_u * div_u_test;

          // Diffusion term:
          res += nu * scalar_product(sym_grad_u, sym_grad_u_test);

          // Pressure term:
          res -= p * div_u_test;

          // Incompressible constraint:
          res -= div_u * p_test;

          // -> Kinetic Energy:
          //////////////////////////////

          double u_norm = std::sqrt(SacadoTools::to_double(ue*ue));

          res += // LHS
            rho * k_dot * k_test;
          if (u_norm > 0 && SUPG_alpha > 0)
            res += ue * grad_k * (k_test + SUPG_alpha * (h/u_norm) * grad_k_test  * ue);
          else
            res += ue * grad_k * k_test;

          res -= // RHS
            + scalar_product(tau, grad_ue) * k_test
            - beta_star * k * we * k_test
            - (nu + sigma_star*k/we) * scalar_product(grad_k, grad_k_test);

          // -> Turbulenc Frequency:
          //////////////////////////////

          double sigma_d = scalar_product(grad_ke, grad_we) <= 0 ? 0 : 1./8.;

          res += // LHS
            rho * w_dot * w_test;
          if (u_norm > 0 && SUPG_alpha > 0)
            res += ue * grad_w * (w_test + SUPG_alpha * (h/u_norm) * grad_w_test  * ue);
          else
            res += ue * grad_w * w_test;

          res -= // RHS
            + alpha * (w/ke) * scalar_product(tau, grad_ue) * w_test
            - beta * w * w * w_test
            + sigma_d / w * scalar_product(grad_ke, grad_we)
            - (nu + sigma*ke/w) * scalar_product(grad_w, grad_w_test);


          residual[0][i] += res * JxW[q];

          //////////////////////
          // PRECONDITIONERS: //
          //////////////////////
          residual[1][i] += p * p_test * JxW[q];
          residual[2][i] += grad_p * grad_p_test * JxW[q];
        }
    }
  (void)compute_only_system_terms;
}


template <int dim, int spacedim, typename LAC>
void
KOmega<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
                                                   LinearOperator<LATrilinos::VectorType> &system_op,
                                                   LinearOperator<LATrilinos::VectorType> &prec_op,
                                                   LinearOperator<LATrilinos::VectorType> &prec_op_finer) const
{
  typedef LATrilinos::VectorType::BlockType  BVEC;
  typedef LATrilinos::VectorType             VEC;

  static ReductionControl solver_control_cg(matrices[0]->m(), CG_solver_tolerance);
  static SolverCG<BVEC > solver_CG(solver_control_cg);

  static ReductionControl solver_control_gmres(matrices[0]->m(), GMRES_solver_tolerance);
  static SolverGMRES<BVEC > solver_GMRES(solver_control_gmres);


  // SYSTEM MATRIX:
  auto A  = linear_operator<BVEC >( matrices[0]->block(0,0) );
  auto Bt = linear_operator<BVEC >( matrices[0]->block(0,1) );
  auto B  = transpose_operator<BVEC >( Bt );
  auto C  = B*Bt;
  auto ZeroP = null_operator(C);

  if (gamma_p!=0.0)
    C = linear_operator<BVEC >( matrices[1]->block(1,1) );
  else
    C = ZeroP;

  ////////////////////////////////////////////////////////////////////////////
  // SYSTEM MATRIX:

  std::array<std::array<LinearOperator<BVEC >, 4>, 4> S, S_pre;
  for (unsigned int i = 0; i<4; ++i)
    for (unsigned int j = 0; j<4; ++j)
      {
        S_pre[i][j] = linear_operator<BVEC >(matrices[0]->block(i,j));
        S[i][j] = null_operator(S_pre[i][j]);
      }

  S[0][0] = A;
  S[0][1] = Bt;
  S[1][0] = B;
  S[1][1] = C;
  S[2][2] = linear_operator<BVEC >(matrices[0]->block(2,2));
  S[3][3] = linear_operator<BVEC >(matrices[0]->block(3,3));

  system_op = BlockLinearOperator<VEC >(S);

  ////////////////////////////////////////////////////////////////////////////
  // PRECONDITIONER MATRIX:

  // D = Diagonal blocks of the system matrix
  // I = Identity operators related to diagonal blocks
  std::array<LinearOperator<BVEC >, 4 > D, I;
  std::array<std::array<LinearOperator<BVEC >, 4 >, 4 > P;

  for (unsigned int i = 0; i<4; ++i)
    {
      D[i] = linear_operator<BVEC >(matrices[0]->block(i,i));
      I[i] = identity_operator(D[i].reinit_range_vector);
      for (unsigned int j = 0; j<i; ++j)
        {
          P[i][j] = linear_operator<BVEC >(matrices[0]->block(i,j));
          P[i][j] = null_operator(P[i][j]);
          P[j][i] = transpose_operator(P[i][j]);
        }
      P[i][i] = I[i];
    }

  const DoFHandler<dim,spacedim> &dh = this->get_dof_handler();
  const ParsedFiniteElement<dim,spacedim> &fe = this->pfe;

  amg_A.initialize_preconditioner<dim, spacedim>( matrices[0]->block(0,0), fe, dh);
  amg_Ap.initialize_preconditioner<dim, spacedim>( matrices[2]->block(1,1), fe, dh);
  amg_k.initialize_preconditioner<dim, spacedim>( matrices[0]->block(2,2), fe, dh);
  amg_w.initialize_preconditioner<dim, spacedim>( matrices[0]->block(3,3), fe, dh);
  jacobi_Mp.initialize_preconditioner( matrices[1]->block(1,1));


  auto alpha = this->get_alpha();
  // auto dt =  1/alpha;

  // Pressure Mass Matrix
  auto Mp = linear_operator<BVEC >(matrices[1]->block(1,1));
  auto Mp_inv = inverse_operator( Mp, solver_CG, jacobi_Mp);

  // Pressure Stiffness Matrix
  auto Ap = linear_operator<BVEC >(matrices[2]->block(1,1));

  auto K = linear_operator<BVEC >(matrices[0]->block(2,2));
  auto W = linear_operator<BVEC >(matrices[0]->block(3,3));

  P[0][1] = -1 * Bt;
  auto BBt  = BlockLinearOperator<VEC >(P);

  ////////////////////////////////////////////////////////////////////////////
  // PRECONDITIONER:
  ////////////////////////////////////////////////////////////////////////////
  auto A_inv = linear_operator<BVEC >(matrices[0]->block(0,0), amg_A);
  auto Ap_inv = linear_operator<BVEC >(matrices[2]->block(1,1), amg_Ap);
  auto Schur_inv = (gamma + nu) * Mp_inv + alpha * rho * Ap_inv;
  auto K_inv = linear_operator<BVEC >(matrices[0]->block(2,2), amg_k);
  auto W_inv = linear_operator<BVEC >(matrices[0]->block(3,3), amg_w);

  typedef LinearOperator<TrilinosWrappers::MPI::Vector,TrilinosWrappers::MPI::Vector> Op_MPI;



  BlockLinearOperator<VEC> D1
  = block_diagonal_operator<4, VEC>(std::array<Op_MPI,4>({{A_inv, I[1], I[2], I[3]}}));

  BlockLinearOperator<VEC> D2 = block_diagonal_operator<4, VEC>(std::array<Op_MPI,4>({{I[0], Schur_inv, K_inv, W_inv}}));


  prec_op = D1 * BBt * D2;

  ////////////////////////////////////////////////////////////////////////////
  // FINER PRECONDITIONER:
  ////////////////////////////////////////////////////////////////////////////
  auto A_inv_finer = inverse_operator(A, solver_GMRES, amg_A);
  auto Ap_inv_finer  = inverse_operator( Ap, solver_CG, amg_Ap);
  auto Schur_inv_finer = (gamma + nu) * Mp_inv + alpha * rho * Ap_inv_finer;
  auto K_inv_finer = inverse_operator(K, solver_GMRES, amg_k);
  auto W_inv_finer = inverse_operator(W, solver_GMRES, amg_w);



  BlockLinearOperator<VEC> D1_finer
  = block_diagonal_operator<4, VEC>(std::array<Op_MPI,4>({{A_inv_finer, I[1], I[2], I[3]}}));

  BlockLinearOperator<VEC> D2_finer = block_diagonal_operator<4, VEC>(std::array<Op_MPI,4>({{I[0], Schur_inv_finer, K_inv_finer, W_inv_finer}}));

  prec_op_finer = D1_finer * BBt * D2_finer;
}

#endif

/*! @} */
