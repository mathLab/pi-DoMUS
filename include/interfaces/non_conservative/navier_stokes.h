#ifndef _dynamic_navier_stokes_h_
#define _dynamic_navier_stokes_h_

#include "interfaces/non_conservative.h"
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

#include <deal2lkit/fe_values_cache.h>

template <int dim>
class NavierStokes : public NonConservativeInterface<dim,dim,dim+1, NavierStokes<dim> >
{
public:
  typedef FEValuesCache<dim,dim> Scratch;
  typedef Assembly::CopyData::piDoMUSPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::piDoMUSSystem<dim,dim> CopySystem;
  typedef TrilinosWrappers::MPI::BlockVector VEC;

  /* specific and useful functions for this very problem */
  NavierStokes();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  /* these functions MUST have the follwowing names
   *  because they are called by the NonConservativeInterface class
   */

  template<typename Number>
  void preconditioner_residual(const typename DoFHandler<dim>::active_cell_iterator &,
                               Scratch &,
                               CopyPreconditioner &,
                               std::vector<Number> &local_residual) const;
  template<typename Number>
  void system_residual(const typename DoFHandler<dim>::active_cell_iterator &,
                       Scratch &,
                       CopySystem &,
                       std::vector<Number> &local_residual) const;



  virtual void compute_system_operators(const DoFHandler<dim> &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        LinearOperator<VEC> &,
                                        LinearOperator<VEC> &) const;

private:
  double eta;
  double rho;

  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

};

template<int dim>
NavierStokes<dim>::NavierStokes() :
  NonConservativeInterface<dim,dim,dim+1,NavierStokes<dim> >("Navier Stokes",
                                                             "FESystem[FE_Q(2)^d-FE_Q(1)]",
                                                             "u,u,p", "1,1; 1,0", "1,0; 0,1","1,0")
{};


template <int dim>
template<typename Number>
void NavierStokes<dim>::preconditioner_residual(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                Scratch &fe_cache,
                                                CopyPreconditioner &data,
                                                std::vector<Number> &local_residual) const
{
  Number alpha = this->alpha;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);

  auto &ps = fe_cache.get_values("solution","p", pressure, alpha);
  auto &grad_us = fe_cache.get_gradients("solution","grad_u", velocity, alpha);
  auto &us = fe_cache.get_values("solution", "u", velocity, alpha);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", velocity, alpha);

  const unsigned int n_q_points = ps.size();

  auto &JxW = fe_cache.get_JxW_values();
  for (unsigned int i=0; i<local_residual.size(); ++i)
    local_residual[i] = 0;
  auto &fev = fe_cache.get_current_fe_values();
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int i=0; i<local_residual.size(); ++i)
        {
          auto v = fev[velocity].value(i,q);
          auto grad_v = fev[velocity].gradient(i,q);
          auto m = fev[pressure].value(i,q);
          const Number &p = ps[q];
          const Tensor<1, dim, Number> &u = us[q];
          const Tensor<1, dim, Number> &u_dot = us_dot[q];
          const Tensor<2, dim, Number> &grad_u = grad_us[q];

          local_residual[i] += (rho*DOFUtilities::inner(u_dot,v)  +
                                eta*DOFUtilities::inner(grad_u,grad_v) +
                                (1./eta)*p*m)*JxW[q];
        }
    }
}

template <int dim>
template<typename Number>
void NavierStokes<dim>::system_residual(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                        Scratch &fe_cache,
                                        CopySystem &data,
                                        std::vector<Number> &local_residual) const
{
  Number alpha = this->alpha;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector velocity(0);
  auto &us = fe_cache.get_values("solution", "u", velocity, alpha);
  auto &div_us = fe_cache.get_divergences("solution", "u", velocity, alpha);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", velocity, alpha);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients("solution", "u", velocity, alpha);
  auto &grad_us = fe_cache.get_gradients("solution", "u", velocity, alpha);

  const FEValuesExtractors::Scalar pressure(dim);
  auto &ps = fe_cache.get_values("solution","p", pressure, alpha);

  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_q_points = ps.size();

  for (unsigned int i=0; i<local_residual.size(); ++i)
    local_residual[i] = 0;
  auto &fev = fe_cache.get_current_fe_values();
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int i=0; i<local_residual.size(); ++i)
        {
          auto v = fev[velocity].value(i,q);
          auto div_v = fev[velocity].divergence(i,q);
          auto grad_v = fev[velocity].gradient(i,q);
          auto m = fev[pressure].value(i,q);

          const Tensor <1, dim, Number> &u = us[q];
          const Tensor <1, dim, Number> &u_dot = us_dot[q];
          const Number &div_u = div_us[q];
          const Number &p = ps[q];
          const Tensor <2, dim, Number> &sym_grad_u = sym_grad_us[q];
          const Tensor <2, dim, Number> &grad_u = grad_us[q];

          local_residual[i] += (rho*DOFUtilities::inner(u_dot,v)
                                + eta*DOFUtilities::inner(grad_u,grad_v)
                                + DOFUtilities::inner((grad_u*u),v)
                                - m*div_u -p*div_v)*JxW[q];
        }
    }
}


template <int dim>
void NavierStokes<dim>::declare_parameters (ParameterHandler &prm)
{
  NonConservativeInterface<dim,dim,dim+1, NavierStokes<dim> >::declare_parameters(prm);
  this->add_parameter(prm, &eta, "eta [Pa s]", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &rho, "rho [Kg m^-d]", "1.0", Patterns::Double(0.0));
}

template <int dim>
void NavierStokes<dim>::parse_parameters_call_back ()
{
  NonConservativeInterface<dim,dim,dim+1, NavierStokes<dim> >::parse_parameters_call_back();
}


template <int dim>
void
NavierStokes<dim>::compute_system_operators(const DoFHandler<dim> &dh,
                                            const TrilinosWrappers::BlockSparseMatrix &matrix,
                                            const TrilinosWrappers::BlockSparseMatrix &preconditioner_matrix,
                                            LinearOperator<VEC> &system_op,
                                            LinearOperator<VEC> &prec_op) const
{

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector velocity_components(0);
  DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(velocity_components),
                                    constant_modes);

  Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.constant_modes = constant_modes;
  Amg_data.elliptic = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 0.02;

  Mp_preconditioner->initialize (preconditioner_matrix.block(1,1));
  Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
                                  Amg_data);


  // SYSTEM MATRIX:
  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,0) );
  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,1) );
  //  auto B =  transpose_operator(Bt);
  auto B     = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,0) );
  auto ZeroP = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,1) );

  auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >( preconditioner_matrix.block(1,1) );

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
  auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);
  auto Schur_inv = inverse_operator( Mp, solver_CG, *Mp_preconditioner);

  auto P00 = A_inv;
  auto P01 = null_operator(Bt);
  auto P10 = Schur_inv * B * A_inv;
  auto P11 = -1 * Schur_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<2, 2, VEC >({{
      {{ A, Bt }} ,
      {{ B, ZeroP }}
    }
  });


  //const auto S = linear_operator<VEC>(matrix);

  prec_op = block_operator<2, 2, VEC >({{
      {{ P00, P01 }} ,
      {{ P10, P11 }}
    }
  });
}


template class NavierStokes <2>;

#endif
