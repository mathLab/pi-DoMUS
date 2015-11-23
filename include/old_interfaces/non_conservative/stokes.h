/*! \addtogroup equations
 *  @{
 */

#ifndef _stokes_nc_h_
#define _stokes_nc_h_

/**
 *  This interface solves a Stokes flow using non conservative interface:
 *  \f[
 *     \begin{cases}
 *    - \textrm{div} \varepsilon(u) + \nabla p = f \\
 *       \textrm{div}u=0
 *     \end{cases}
 *  \f]
 *  where \f$ \varepsilon(u) = \frac{\nabla u + [\nabla u]^t}{2}. \f$
 */

#include "interfaces/non_conservative.h"

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
#include <deal2lkit/parsed_function.h>

template <int dim>
class StokesNC : public NonConservativeInterface<dim,dim,dim+1, StokesNC<dim> >
{
public:
  typedef FEValuesCache<dim,dim> Scratch;
  typedef Assembly::CopyData::piDoMUSPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::piDoMUSSystem<dim,dim> CopySystem;
  typedef TrilinosWrappers::MPI::BlockVector VEC;
  typedef TrilinosWrappers::BlockSparseMatrix MAT;

  /* specific and useful functions for this very problem */
  StokesNC();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

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
                                        const std::vector<shared_ptr<MAT> >,
                                        LinearOperator<VEC> &,
                                        LinearOperator<VEC> &) const;

  template<typename Number>
  void aux_matrix_residuals(const typename DoFHandler<dim>::active_cell_iterator &,
                            Scratch &,
                            CopyPreconditioner &,
                            std::vector<std::vector<Number> > &) const
  {};


private:
  double eta;

  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

};

template<int dim>
StokesNC<dim>::StokesNC() :
  NonConservativeInterface<dim,dim,dim+1,StokesNC<dim> >("StokesNC",
                                                         "FESystem[FE_Q(2)^d-FE_Q(1)]",
                                                         "u,u,p", "1,1; 1,0", "1,0; 0,1","0,0")
{};


template <int dim>
template<typename Number>
void StokesNC<dim>::preconditioner_residual(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                            Scratch &fe_cache,
                                            CopyPreconditioner &data,
                                            std::vector<Number> &local_residual) const
{
  Number alpha = this->alpha;
  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution",      *this->solution,      alpha);
  fe_cache.cache_local_solution_vector("solution_dot",  *this->solution_dot,  alpha);

  this->fix_solution_dot_derivative(fe_cache, alpha);

  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);

  auto &us      = fe_cache.get_values(    "solution",     "u",      velocity,       alpha);
  auto &grad_us = fe_cache.get_gradients( "solution",     "grad_u", velocity,       alpha);
  auto &us_dot  = fe_cache.get_values(    "solution_dot", "u_dot",  velocity,       alpha);
  auto &ps      = fe_cache.get_values(    "solution",     "p",      pressure,       alpha);

  const unsigned int n_q_points = ps.size();

  // Jacobian:
  auto &JxW = fe_cache.get_JxW_values();

  // Init residual to 0
  for (unsigned int i=0; i<local_residual.size(); ++i)
    local_residual[i] = 0;

  auto &fev = fe_cache.get_current_fe_values();
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int i=0; i<local_residual.size(); ++i)
        {
          // test functions:
          auto v      = fev[velocity    ].value(i,q);
          auto grad_v = fev[velocity    ].gradient(i,q);
          auto m      = fev[pressure    ].value(i,q);

          // variables:
          const Tensor<1, dim, Number> &u       = us[q];
          const Tensor<1, dim, Number> &u_dot   = us_dot[q];
          const Tensor<2, dim, Number> &grad_u  = grad_us[q];
          const Number                 &p       = ps[q];

          // compute residual:
          local_residual[i] +=
            (eta * scalar_product(grad_u,grad_v) +
             (1./eta)*p*m)*JxW[q];

        }
    }
}


template <int dim>
template<typename Number>
void
StokesNC<dim>::
system_residual(const typename DoFHandler<dim>::active_cell_iterator &cell,
                Scratch &fe_cache,
                CopySystem &data,
                std::vector<Number> &local_residual) const
{
  Number alpha = this->alpha;
  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution",      *this->solution,      alpha);
  fe_cache.cache_local_solution_vector("solution_dot",  *this->solution_dot,  alpha);

  this->fix_solution_dot_derivative(fe_cache, alpha);

  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);


  // velocity:
  auto &us          = fe_cache.get_values(                "solution",     "u",      velocity,       alpha);
  auto &grad_us     = fe_cache.get_gradients(             "solution",     "grad_u", velocity,       alpha);
  auto &div_us      = fe_cache.get_divergences(           "solution",     "div_u",  velocity,       alpha);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients(   "solution",     "u",      velocity,       alpha);
  auto &us_dot      = fe_cache.get_values(                "solution_dot", "u_dot",  velocity,       alpha);

  // pressure:
  auto &ps          = fe_cache.get_values(                "solution",     "p",      pressure,       alpha);

  // Jacobian:
  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_q_points = ps.size();

  // Init residual to 0
  for (unsigned int i=0; i<local_residual.size(); ++i)
    local_residual[i] = 0;

  auto &fev = fe_cache.get_current_fe_values();
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int i=0; i<local_residual.size(); ++i)
        {
          // test functions:
          //    velocity:
          auto v      = fev[velocity    ].value(i,q);
          auto grad_v = fev[velocity    ].gradient(i,q);
          auto div_v  = fev[velocity    ].divergence(i,q);

          //    pressure:
          auto m      = fev[pressure    ].value(i,q);

          // variables:
          //    velocity:
          const Tensor<1, dim, Number>  &u          = us[q];
          const Number                  &div_u      = div_us[q];
          const Tensor<1, dim, Number>  &u_dot      = us_dot[q];
          const Tensor<2, dim, Number>  &grad_u     = grad_us[q];
          const Tensor <2, dim, Number> &sym_grad_u = sym_grad_us[q];

          //    pressure:
          const Number                  &p          = ps[q];

          local_residual[i] += ( eta*scalar_product( sym_grad_u , grad_v )
                                 + p * div_v  + m * div_u )
                               * JxW[q];
        }
    }
}




template <int dim>
void
StokesNC<dim>::
declare_parameters (ParameterHandler &prm)
{
  NonConservativeInterface<dim,dim,dim+1, StokesNC<dim> >::declare_parameters(prm);
  this->add_parameter(prm, &eta, "eta [Pa s]", "1.0", Patterns::Double(0.0));
}

template <int dim>
void
StokesNC<dim>::
parse_parameters_call_back ()
{
  NonConservativeInterface<dim,dim,dim+1, StokesNC<dim> >::parse_parameters_call_back();
}


template <int dim>
void
StokesNC<dim>::compute_system_operators(const DoFHandler<dim> &dh,
                                        const TrilinosWrappers::BlockSparseMatrix &matrix,
                                        const TrilinosWrappers::BlockSparseMatrix &preconditioner_matrix,
                                        const std::vector<shared_ptr<MAT> >,
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


template class StokesNC <2>;

#endif
/*! @} */
