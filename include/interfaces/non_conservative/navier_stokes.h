/*! \addtogroup equations
 *  @{
 */

#ifndef _navier_stokes_h_
#define _navier_stokes_h_

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
 *
 * In the code we adopt the following notations:
 * - Mp := block resulting from \f$ ( \partial_t p, q ) \f$
 * - Ap := block resulting from f$ \nu ( \nabla p,\nabla q ) \f$
 * - Np := block resulting from f$ ( u \cdot \nabla p, q) \f$
 * - Fp := Mp + Ap + Np
 *
 * where:
 * - p = pressure
 * - q = test function for the pressure
 * - u = velocity
 * - v = test function for the velocity
 */

#include "interfaces/non_conservative.h"

#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal2lkit/parsed_function.h>
#include <deal2lkit/fe_values_cache.h>
#include <deal2lkit/parsed_function.h>


template <int dim>
class NavierStokes : public NonConservativeInterface<dim,dim,dim+1, NavierStokes<dim> >
{
public:
  typedef FEValuesCache<dim,dim> Scratch;

  typedef Assembly::CopyData::piDoMUSPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::piDoMUSSystem<dim,dim> CopySystem;

  // Types:
  typedef TrilinosWrappers::MPI::BlockVector BVEC;
  typedef TrilinosWrappers::MPI::Vector VEC;
  typedef TrilinosWrappers::BlockSparseMatrix MAT;

  NavierStokes();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  template<typename Number>
  void preconditioner_residual(
    const typename DoFHandler<dim>::active_cell_iterator &,
    Scratch &,
    CopyPreconditioner &,
    std::vector<Number> &local_residual) const;

  template<typename Number>
  void system_residual(
    const typename DoFHandler<dim>::active_cell_iterator &,
    Scratch &,
    CopySystem &,
    std::vector<Number> &local_residual) const;

  template<typename Number>
  void
  aux_matrix_residuals(
    const typename DoFHandler<dim>::active_cell_iterator &,
    Scratch &,
    CopyPreconditioner &,
    std::vector<std::vector<Number> > &local_residual) const;

  /**
   * specify the number of auxiliry matrices that the problem requires.
   * @return number of auxiliary matrices.
   */
  virtual unsigned int get_number_of_aux_matrices() const
  {
    return 4; // Ap Mp Np Fp
  }

  virtual void compute_system_operators(const DoFHandler<dim> &,
                                        const MAT &,
                                        const MAT &,
                                        const std::vector<shared_ptr<MAT> >,
                                        LinearOperator<BVEC> &,
                                        LinearOperator<BVEC> &) const;

private:

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
   * AMG high order:
   */
  bool amg_higher_order;

  /**
   * AMG smoother sweeps:
   */
  int amg_smoother_sweeps;

  /**
   * AMG aggregation threshold:
   */
  double amg_aggregation_threshold;

  /**
   * AMG smoother sweeps:
   */
  int amg_p_smoother_sweeps;

  /**
   * AMG aggregation threshold:
   */
  double amg_p_aggregation_threshold;

  /**
   * Name of the preconditioner:
   */
  std::string prec_name;

  /**
   * Invert Mp using inverse_operator
   */
  bool invert_Mp;

  /**
   * Invert Np using inverse_operator
   */
  bool invert_Np;

  /**
   * Invert Ap using inverse_operator
   */
  bool invert_Ap;

  /**
   * Invert Fp using inverse_operator
   */
  bool invert_Fp;

  /**
   * Solver tolerance for CG
   */
  double CG_solver_tolerance;

  /**
   * Solver tolerance for GMRES
   */
  double GMRES_solver_tolerance;

  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>  Amg_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>  Amg_preconditioner_2;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> jacobi_preconditioner;

};

template<int dim>
NavierStokes<dim>::
NavierStokes()
  :
  NonConservativeInterface<dim,dim,dim+1,NavierStokes<dim> >("Navier Stokes",
                                                             "FESystem[FE_Q(2)^d-FE_Q(1)]",
                                                             "u,u,p",
                                                             "1,1; 1,0",
                                                             "0,0; 0,1",
                                                             "1,0")
{}

template <int dim>
void
NavierStokes<dim>::
declare_parameters (ParameterHandler &prm)
{
  NonConservativeInterface<dim,dim,dim+1, NavierStokes<dim> >::declare_parameters(prm);
  this->add_parameter(prm, &rho,
                      "rho [kg m^3]", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &nu,
                      "nu [Pa s]", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &gamma,
                      "grad-div stabilization", "1.0", Patterns::Double());
  this->add_parameter(prm, &prec_name,  "Preconditioner","stokes",
                      Patterns::Selection("stokes|low-nu|elman-1|elman-2|BFBt_id|BFBt_dA|cah-cha"),
                      "Available preconditioners: \n"
                      " - stokes  -> S^-1 = 1/nu * Mp^-1 \n"
                      " - low-nu  -> S^-1 = rho * alpha * Mp^-1 \n"
                      " - elman-1 -> S^-1 = Ap (BBt)^-1 \n"
                      " - elman-2 -> S^-1 = Ap Fp^-1 Mp \n"
                      " - BFBt_id -> S^-1 = (BBt)^-1 B A Bt (BBt)^-1\n"
                      " - BFBt_dA -> S^-1 = (B diag(A)^-1 Bt)^-1 B diag(A)^-1 A diag(A)^-1 Bt (B diag(A)^-1 Bt)^-1\n"
                      " - cah-cha -> S^-1 = Mp^-1 + Ap^-1\n");
  this->add_parameter(prm, &amg_higher_order,
                      "Amg Higher Order", "true", Patterns::Bool());
  this->add_parameter(prm, &amg_smoother_sweeps,
                      "Amg Smoother Sweeps", "2", Patterns::Integer(0));
  this->add_parameter(prm, &amg_aggregation_threshold,
                      "Amg Aggregation Threshold", "0.02", Patterns::Double(0.0));
  this->add_parameter(prm, &amg_p_smoother_sweeps,
                      "Amg P Smoother Sweeps","2", Patterns::Integer(0));
  this->add_parameter(prm, &amg_p_aggregation_threshold,
                      "Amg P Aggregation Threshold", "0.02", Patterns::Double(0.0));
  this->add_parameter(prm, &invert_Ap,
                      "Invert Ap using inverse_operator", "true", Patterns::Bool());
  this->add_parameter(prm, &invert_Mp,
                      "Invert Mp using inverse_operator", "true", Patterns::Bool());
  this->add_parameter(prm, &invert_Fp,
                      "Invert Fp using inverse_operator", "true", Patterns::Bool());
  this->add_parameter(prm, &invert_Np,
                      "Invert Np using inverse_operator", "true", Patterns::Bool());
  this->add_parameter(prm, &CG_solver_tolerance,
                      "CG Solver tolerance", "1e-8",
                      Patterns::Double(0.0));
  this->add_parameter(prm, &GMRES_solver_tolerance,
                      "GMRES Solver tolerance", "1e-8",
                      Patterns::Double(0.0));
}

template <int dim>
void
NavierStokes<dim>::
parse_parameters_call_back ()
{
  NonConservativeInterface<dim,dim,dim+1, NavierStokes<dim> >::parse_parameters_call_back();
}

template <int dim>
template<typename Number>
void
NavierStokes<dim>::
preconditioner_residual(const typename DoFHandler<dim>::active_cell_iterator &cell,
                        Scratch &fe_cache,
                        CopyPreconditioner &data,
                        std::vector<Number> &local_residual) const
{
  Number alpha = this->alpha;
  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
  fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);

  this->fix_solution_dot_derivative(fe_cache, alpha);

  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);

  auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);
  auto &ps_dot = fe_cache.get_values("solution_dot", "p_dot", pressure, alpha);
  auto &grad_ps = fe_cache.get_gradients("solution", "grad_p", pressure, alpha);
  auto &us = fe_cache.get_values("solution", "u", velocity, alpha);

  const unsigned int n_q_points = ps.size();

// Jacobian:
  auto &JxW = fe_cache.get_JxW_values();

// Init residual to 0
  for (unsigned int i=0; i<local_residual.size(); ++i)
    local_residual[i] = 0;

  auto &fev = fe_cache.get_current_fe_values();
  for (unsigned int q=0; q<n_q_points; ++q)
    {
// variables:
      const Number &p = ps[q];
      const Number &p_dot = ps_dot[q];
      const Tensor<1, dim, Number> &grad_p = grad_ps[q];
      const Tensor<1, dim, Number> &u = us[q];

      for (unsigned int i=0; i<local_residual.size(); ++i)
        {
// test functions:
          auto m = fev[ pressure ].value(i,q);
          auto grad_m = fev[ pressure ].gradient(i,q);

// compute residual:
          local_residual[i] += 0;
        }

    }

}

template <int dim>
template<typename Number>
void
NavierStokes<dim>::
system_residual(const typename DoFHandler<dim>::active_cell_iterator &cell,
                Scratch &fe_cache,
                CopySystem &data,
                std::vector<Number> &local_residual) const
{
  Number alpha = this->alpha;
  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
  fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);

  this->fix_solution_dot_derivative(fe_cache, alpha);

  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);


// velocity:
  auto &us = fe_cache.get_values("solution", "u", velocity, alpha);
  auto &div_us = fe_cache.get_divergences("solution", "div_u", velocity, alpha);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients("solution", "sym_grad_u", velocity, alpha);
  auto &grad_us = fe_cache.get_gradients("solution", "grad_u", velocity, alpha);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", velocity, alpha);

// pressure:
  auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);
  auto &grad_ps = fe_cache.get_gradients("solution", "p", pressure, alpha);

// Jacobian:
  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_q_points = ps.size();

// Init residual to 0
  for (unsigned int i=0; i<local_residual.size(); ++i)
    local_residual[i] = 0;

  auto &fev = fe_cache.get_current_fe_values();
  for (unsigned int q=0; q<n_q_points; ++q)
    {
// variables:
// velocity:
      const Tensor<1, dim, Number> &u = us[q];
      const Number &div_u = div_us[q];
      const Tensor<1, dim, Number> &u_dot = us_dot[q];
      const Tensor <2, dim, Number> &sym_grad_u = sym_grad_us[q];
      const Tensor <2, dim, Number> &grad_u = grad_us[q];

// pressure:
      const Number &p = ps[q];
      const Tensor<1, dim, Number> &grad_p = grad_ps[q];

      for (unsigned int i=0; i<local_residual.size(); ++i)
        {
// test functions:
// velocity:
          auto v = fev[velocity ].value(i,q);
          auto grad_v = fev[velocity ].gradient(i,q);
          auto sym_grad_v = fev[velocity].symmetric_gradient(i,q);
          auto div_v = fev[velocity ].divergence(i,q);

// pressure:
          auto m = fev[pressure ].value(i,q);
          auto grad_m = fev[pressure ].gradient(i,q);

// compute residual:
          local_residual[i] += (
                                 rho * u_dot * v +
                                 rho * scalar_product(u*grad_u, v) +
                                 gamma * div_u * div_v +
                                 nu * scalar_product(sym_grad_u,sym_grad_v) -
                                 ( p * div_v + div_u * m)
                               )*JxW[q];
        }
    }
}

template <int dim>
template<typename Number>
void
NavierStokes<dim>::
aux_matrix_residuals(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     Scratch &fe_cache,
                     CopyPreconditioner &data,
                     std::vector<std::vector<Number> > &local_residuals) const
{
  Number alpha = this->alpha;
  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
  fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);

  this->fix_solution_dot_derivative(fe_cache, alpha);

  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);

  auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);
  auto &ps_dot = fe_cache.get_values("solution_dot", "p_dot", pressure, alpha);
  auto &grad_ps = fe_cache.get_gradients("solution", "grad_p", pressure, alpha);
  auto &us = fe_cache.get_values("solution", "u", velocity, alpha);
  auto &div_us = fe_cache.get_divergences("solution", "div_u", velocity, alpha);

  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_q_points = ps.size();

  auto &fev = fe_cache.get_current_fe_values();
// Init residual to 0
  for (unsigned int aux=0; aux<get_number_of_aux_matrices(); ++aux)
    for (unsigned int i=0; i<local_residuals[0].size(); ++i)
      local_residuals[aux][i] = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      // variables:
      const Number &p = ps[q];
      const Number &p_dot = ps_dot[q];
      const Tensor<1, dim, Number> &grad_p = grad_ps[q];
      const Tensor<1, dim, Number> &u = us[q];
      const Number &div_u = div_us[q];
      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
          // test functions:
          auto v = fev[velocity ].value(i,q);
          auto div_v = fev[velocity ].divergence(i,q);
          auto m = fev[ pressure ].value(i,q);
          auto grad_m = fev[ pressure ].gradient(i,q);


// compute residuals:
          local_residuals[0][i] += ( // Ap
                                     scalar_product( grad_p,grad_m )
                                   )*JxW[q];

          local_residuals[1][i] += ( // Np
                                     scalar_product( u,grad_p) * m +
                                     gamma * div_u * div_v
                                   )*JxW[q];

          local_residuals[2][i] += ( // Mp
                                     m * p
                                   )*JxW[q];

          local_residuals[3][i] += ( // Fp
                                     nu * scalar_product( grad_p,grad_m ) +
                                     scalar_product( u,grad_p) * m +
                                     gamma * div_u * div_v +
                                     alpha * m * p
                                   )*JxW[q];
        }
    }
}

template <int dim>
void
NavierStokes<dim>::
compute_system_operators(const DoFHandler<dim> &dh,
                         const MAT &matrix,
                         const MAT &preconditioner_matrix,
                         const std::vector<shared_ptr<MAT> > aux_matrices,
                         LinearOperator<BVEC> &system_op,
                         LinearOperator<BVEC> &prec_op) const
{
  auto alpha = this->alpha;

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector velocity_components(0);
  DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(velocity_components),
                                    constant_modes);
  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.constant_modes = constant_modes;
// Amg_data.elliptic = true;
  Amg_data.higher_order_elements = amg_higher_order;
  Amg_data.smoother_sweeps = amg_smoother_sweeps;
  Amg_data.aggregation_threshold = amg_aggregation_threshold;

  std::vector<std::vector<bool> > constant_modes_p;
  FEValuesExtractors::Scalar pressure_components(dim);
  DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(pressure_components),
                                    constant_modes_p);
  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_p;
  Amg_data_p.constant_modes = constant_modes_p;
  Amg_data_p.elliptic = false;
  Amg_data_p.smoother_sweeps = amg_p_smoother_sweeps;
  Amg_data_p.aggregation_threshold = amg_p_aggregation_threshold;

// SYSTEM MATRIX:
  auto A = linear_operator< VEC >( matrix.block(0,0) );
  auto Bt = linear_operator< VEC >( matrix.block(0,1) );
  auto B = transpose_operator(Bt);
  auto C = linear_operator< VEC >(aux_matrices[0]->block(1,1));
  auto ZeroP = null_operator(C);

  system_op = block_operator<2, 2, BVEC >({{
      {{ A, Bt }} ,
      {{ B, ZeroP }}
    }
  });

// PRECONDITIONER

  static ReductionControl solver_control_cg(matrix.m(), CG_solver_tolerance);
  static SolverCG<VEC> solver_CG(solver_control_cg);

  static ReductionControl solver_control_gmres(matrix.m(), GMRES_solver_tolerance);
  static SolverGMRES<VEC> solver_GMRES(solver_control_gmres);


  Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
  Amg_preconditioner->initialize (matrix.block(0,0), Amg_data);
  auto A_inv = inverse_operator( A, solver_GMRES, *Amg_preconditioner);

  LinearOperator<VEC> P00, P01, P10, P11, Schur_inv;

  auto Ap= linear_operator<VEC>(aux_matrices[0]->block(1,1));
  auto Np= linear_operator<VEC>(aux_matrices[2]->block(1,1));
  auto Mp= linear_operator<VEC>(aux_matrices[2]->block(1,1));
  auto Fp = linear_operator<VEC>(aux_matrices[3]->block(1,1));

  Assert(prec_name != "", ExcNotInitialized());
  if (prec_name=="stokes")
    {
      jacobi_preconditioner.reset (new TrilinosWrappers::PreconditionJacobi());
      jacobi_preconditioner->initialize (aux_matrices[2]->block(1,1),1.3);
      auto Mp_inv = inverse_operator( Mp, solver_CG, *jacobi_preconditioner);

      Schur_inv = 1/nu * Mp_inv;
    }
  else if (prec_name=="low-nu")
    {
      Amg_preconditioner_2.reset (new TrilinosWrappers::PreconditionAMG());
      Amg_preconditioner_2->initialize (aux_matrices[0]->block(1,1),  Amg_data_p);
      LinearOperator<VEC> Ap_inv;
      if (invert_Ap)
        {
          Ap_inv  = inverse_operator( Ap, solver_CG, *Amg_preconditioner_2);
        }
      else
        {
          Ap_inv = linear_operator<VEC>(aux_matrices[0]->block(1,1), *Amg_preconditioner_2);
        }

      Schur_inv = rho * alpha * Ap_inv;
    }
  else if (prec_name=="cah-cha")
    {
      Amg_preconditioner_2.reset (new TrilinosWrappers::PreconditionAMG());
      Amg_preconditioner_2->initialize (aux_matrices[1]->block(1,1),  Amg_data_p);
      LinearOperator<VEC> Ap_inv;
      if (invert_Ap)
        {
          Ap_inv  = inverse_operator( Ap, solver_CG, *Amg_preconditioner_2);
        }
      else
        {
          Ap_inv = linear_operator<VEC>(aux_matrices[1]->block(1,1), *Amg_preconditioner_2);
        }

      jacobi_preconditioner.reset (new TrilinosWrappers::PreconditionJacobi());
      jacobi_preconditioner->initialize (aux_matrices[2]->block(1,1));
      auto Mp = linear_operator<VEC>( aux_matrices[2]->block(1,1) );
      auto Mp_inv  = inverse_operator( Mp, solver_CG, *jacobi_preconditioner);

      Schur_inv = Mp_inv + Ap_inv;
    }
  else if (prec_name=="BFBt_id")
    {
      auto BBt = B*Bt;
      Amg_preconditioner_2.reset (new TrilinosWrappers::PreconditionAMG());
      Amg_preconditioner_2->initialize (aux_matrices[0]->block(1,1), Amg_data_p);
      LinearOperator<VEC> BBt_inv;
      if (invert_Ap)
        {
          BBt_inv  = inverse_operator( BBt, solver_CG, *Amg_preconditioner_2);
        }
      else
        {
          BBt_inv = linear_operator<VEC>(aux_matrices[0]->block(1,1), *Amg_preconditioner_2);
        }

      Schur_inv = BBt_inv * B * A * Bt * BBt_inv;
    }
  else if (prec_name=="BFBt_dA")
    {
      Amg_preconditioner_2.reset (new TrilinosWrappers::PreconditionAMG());
      Amg_preconditioner_2->initialize (aux_matrices[0]->block(1,1), Amg_data_p);

      auto inv_diag_A = linear_operator< VEC >( matrix.block(0,0) );
      inv_diag_A.vmult = [&matrix](VEC &v, const VEC &u)
      {
        for (auto i : v.locally_owned_elements())
          v(i)=u(i)/matrix.block(0,0)(i,i);
      };

      auto BBt = B*inv_diag_A*Bt;
      auto BBt_inv  = inverse_operator( BBt, solver_CG, *Amg_preconditioner_2);

      Schur_inv = BBt_inv * B * inv_diag_A *A * inv_diag_A * Bt * BBt_inv;
    }
  else if (prec_name=="elman-1")
    {
      auto BBt = B*Bt;
      Amg_preconditioner_2.reset (new TrilinosWrappers::PreconditionAMG());
      Amg_preconditioner_2->initialize (aux_matrices[0]->block(1,1), Amg_data_p);
      LinearOperator<VEC> BBt_inv;
      if (invert_Ap)
        {
          BBt_inv  = inverse_operator( BBt, solver_CG, *Amg_preconditioner_2);
        }
      else
        {
          BBt_inv = linear_operator<VEC>(aux_matrices[1]->block(1,1), *Amg_preconditioner_2);
        }

      Schur_inv = Ap*BBt_inv;
    }
  else if (prec_name=="elman-2")
    {
      Amg_preconditioner_2.reset (new TrilinosWrappers::PreconditionAMG());
      Amg_preconditioner_2->initialize( aux_matrices[3]->block(1,1), Amg_data_p);
      LinearOperator<VEC> Fp_inv;
      if (invert_Fp)
        {
          Fp_inv = inverse_operator( Fp, solver_GMRES, *Amg_preconditioner_2);
        }
      else
        {
          Fp_inv = linear_operator<VEC>(aux_matrices[3]->block(1,1), *Amg_preconditioner_2 );
        }

      Schur_inv = Ap * Fp_inv * Mp;
    }
  else
    {
      AssertThrow(false, ExcMessage("Preconditioner not recognized."));
    }

  P00 = A_inv;
  P01 = null_operator(Bt);
  P10 = Schur_inv * B * A_inv;
  P11 = -1 * Schur_inv;

  prec_op = block_operator<2, 2, BVEC >({{
      {{ P00, P01 }} ,
      {{ P10, P11 }}
    }
  });

}

template class NavierStokes <2>;
template class NavierStokes <3>;

#endif
/*! @} */
