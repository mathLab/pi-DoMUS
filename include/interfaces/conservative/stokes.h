/*! \addtogroup equations
 *  @{
 */

#ifndef _stokes_h_
#define _stokes_h_

/**
 *  This interface solves a Stokes flow:
 *  \f[
 *     \begin{cases}
 *    - \textrm{div} \varepsilon(u) + \nabla p = f \\
 *       \textrm{div}u=0
 *     \end{cases}
 *  \f]
 *  where \f$ \varepsilon(u) = \frac{\nabla u + [\nabla u]^t}{2}. \f$
 */

#include "interfaces/conservative.h"
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
class Stokes : public ConservativeInterface<dim,dim,dim+1, Stokes<dim> >
{
public:
  typedef FEValuesCache<dim,dim> Scratch;
  typedef Assembly::CopyData::piDoMUSPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::piDoMUSSystem<dim,dim> CopySystem;
  typedef TrilinosWrappers::MPI::BlockVector VEC;
  typedef TrilinosWrappers::BlockSparseMatrix MAT;

  /* specific and useful functions for this very problem */
  Stokes();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  /* these functions MUST have the follwowing names
   *  because they are called by the ConservativeInterface class
   */

  template<typename Number>
  void preconditioner_energy(const typename DoFHandler<dim>::active_cell_iterator &,
                             Scratch &,
                             CopyPreconditioner &,
                             Number &energy) const;

  template<typename Number>
  void system_energy(const typename DoFHandler<dim>::active_cell_iterator &,
                     Scratch &,
                     CopySystem &,
                     Number &energy) const;

  virtual void compute_system_operators(const DoFHandler<dim> &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        const std::vector<shared_ptr<MAT> >,
                                        LinearOperator<VEC> &,
                                        LinearOperator<VEC> &) const;

private:
  double eta;
  bool block_back_substitution_bool;

  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

};

template<int dim>
Stokes<dim>::Stokes() :
  ConservativeInterface<dim,dim,dim+1,Stokes<dim> >("Stokes",
                                                    "FESystem[FE_Q(2)^d-FE_Q(1)]",
                                                    "u,u,p", "1,1; 1,0", "1,0; 0,1","0,0")
{};


template <int dim>
template<typename Number>
void Stokes<dim>::preconditioner_energy(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                        Scratch &fe_cache,
                                        CopyPreconditioner &data,
                                        Number &energy) const
{
  Number alpha = this->alpha;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);
  auto &ps = fe_cache.get_values("solution","p", pressure, alpha);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients("solution","sym_grad_u", velocity, alpha);
  auto &us = fe_cache.get_values("solution", "u", velocity, alpha);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", velocity, alpha);

  const unsigned int n_q_points = ps.size();

  auto &JxW = fe_cache.get_JxW_values();
  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Number &p = ps[q];
      const Tensor<1, dim, Number> &u = us[q];
      const Tensor<1, dim, Number> &u_dot = us_dot[q];
      const Tensor<2, dim, Number> &sym_grad_u = sym_grad_us[q];

      energy += (eta*.5*scalar_product(sym_grad_u,sym_grad_u) +
                 (1./eta)*p*p)*JxW[q];
    }
}

template <int dim>
template<typename Number>
void Stokes<dim>::system_energy(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                Scratch &fe_cache,
                                CopySystem &data,
                                Number &energy) const
{
  Number alpha = this->alpha;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector velocity(0);
  auto &us = fe_cache.get_values("solution", "u", velocity, alpha);
  auto &div_us = fe_cache.get_divergences("solution", "u", velocity, alpha);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", velocity, alpha);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients("solution", "u", velocity, alpha);

  const FEValuesExtractors::Scalar pressure(dim);
  auto &ps = fe_cache.get_values("solution","p", pressure, alpha);

  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_q_points = ps.size();

  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {

      const Tensor <1, dim, Number> &u = us[q];
      const Tensor <1, dim, Number> &u_dot = us_dot[q];
      const Number &div_u = div_us[q];
      const Number &p = ps[q];
      const Tensor <2, dim, Number> &sym_grad_u = sym_grad_us[q];

      Number psi = eta*.5*scalar_product(sym_grad_u,sym_grad_u) - p*div_u;
      energy += psi*JxW[q];
    }
}


template <int dim>
void Stokes<dim>::declare_parameters (ParameterHandler &prm)
{
  ConservativeInterface<dim,dim,dim+1, Stokes<dim> >::declare_parameters(prm);
  this->add_parameter(prm, &eta, "eta [Pa s]", "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &block_back_substitution_bool, "use block_back_substitution", "false", Patterns::Bool());
}

template <int dim>
void Stokes<dim>::parse_parameters_call_back ()
{
  ConservativeInterface<dim,dim,dim+1, Stokes<dim> >::parse_parameters_call_back();
}


template <int dim>
void
Stokes<dim>::compute_system_operators(const DoFHandler<dim> &dh,
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

  if (block_back_substitution_bool)
    {
      const auto Mat = block_operator<2, 2, VEC >({{
          {{ A, Bt }} ,
          {{ B, ZeroP }}
        }
      });

      const auto DiagInv = block_operator<2, 2, VEC >({{
          {{ A_inv, Bt }} ,
          {{ B, -1*Schur_inv }}
        }
      } );

      prec_op = block_back_substitution<VEC>( Mat, DiagInv );
      system_op  = Mat;


    }
  else
    {
      auto P00 = A_inv;
      auto P01 = null_operator(Bt);
      auto P10 = Schur_inv * B * A_inv;
      auto P11 = -1 * Schur_inv;
      system_op  = block_operator<2, 2, VEC >({{
          {{ A, Bt }} ,
          {{ B, ZeroP }}
        }
      });
      prec_op = block_operator<2, 2, VEC >({{
          {{ P00, P01 }} ,
          {{ P10, P11 }}
        }
      });
    }
}


template class Stokes <2>;

#endif
/*! @} */
