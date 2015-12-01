/*! \addtogroup equations
 *  @{
 */

#ifndef _dynamic_navier_stokes_h_
#define _dynamic_navier_stokes_h_

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

// for DOFUtilities::inner
using namespace DOFUtilities;

template <int dim>
class ALENavierStokes : public NonConservativeInterface<dim,dim,dim+dim+1, ALENavierStokes<dim> >
{
public:
  typedef FEValuesCache<dim,dim> Scratch;
  typedef Assembly::CopyData::piDoMUSPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::piDoMUSSystem<dim,dim> CopySystem;
  typedef TrilinosWrappers::MPI::BlockVector VEC;
  typedef TrilinosWrappers::BlockSparseMatrix MAT;

  /* specific and useful functions for this very problem */
  ALENavierStokes();

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
  double rho;

  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> P00_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    P11_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> P22_preconditioner;
};

template<int dim>
ALENavierStokes<dim>::ALENavierStokes() :
  NonConservativeInterface<dim,dim,dim+dim+1,ALENavierStokes<dim> >("ALE Navier Stokes",
      "FESystem[FE_Q(2)^d-FE_Q(2)^d-FE_Q(1)]",
      "d,d,u,u,p",
      "1,0,0; 1,1,1; 0,0,1",
      "1,0,0; 0,1,0; 0,0,1",
      "1,1,0")
{};


template <int dim>
template<typename Number>
void ALENavierStokes<dim>::preconditioner_residual(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                   Scratch &fe_cache,
                                                   CopyPreconditioner &data,
                                                   std::vector<Number> &local_residual) const
{
  Number alpha = this->alpha;
  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution",      *this->solution,      alpha);
  fe_cache.cache_local_solution_vector("solution_dot",  *this->solution_dot,  alpha);

  this->fix_solution_dot_derivative(fe_cache, alpha);

  const FEValuesExtractors::Vector velocity(dim);
  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Scalar pressure(2*dim);

  auto &us      = fe_cache.get_values(    "solution",     "u",      velocity,       alpha);
  auto &grad_us = fe_cache.get_gradients( "solution",     "grad_u", velocity,       alpha);
  auto &us_dot  = fe_cache.get_values(    "solution_dot", "u_dot",  velocity,       alpha);
  auto &grad_ds = fe_cache.get_gradients( "solution",     "grad_d", displacement,   alpha);
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
          auto grad_e = fev[displacement].gradient(i,q);
          auto m      = fev[pressure    ].value(i,q);

          // variables:
          const Tensor<1, dim, Number> &u       = us[q];
          const Tensor<1, dim, Number> &u_dot   = us_dot[q];
          const Tensor<2, dim, Number> &grad_u  = grad_us[q];
          const Tensor<2, dim, Number> &grad_d  = grad_ds[q];
          const Number                 &p       = ps[q];

          // compute residual:
          local_residual[i] += (rho * inner(u_dot,v)  +
                                eta * inner(grad_u,grad_v) +
                                inner(grad_d,grad_e) +
                                (1./eta)*p*m)*JxW[q];
        }
    }
}

template <int dim>
template<typename Number>
void ALENavierStokes<dim>::system_residual(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                           Scratch &fe_cache,
                                           CopySystem &data,
                                           std::vector<Number> &local_residual) const
{
  Number alpha = this->alpha;
  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution",      *this->solution,      alpha);
  fe_cache.cache_local_solution_vector("solution_dot",  *this->solution_dot,  alpha);

  this->fix_solution_dot_derivative(fe_cache, alpha);

  const FEValuesExtractors::Vector velocity(dim);
  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Scalar pressure(2*dim);


  // velocity:
  auto &us          = fe_cache.get_values(                "solution",     "u",      velocity,       alpha);
  auto &grad_us     = fe_cache.get_gradients(             "solution",     "grad_u", velocity,       alpha);
  auto &div_us      = fe_cache.get_divergences(           "solution",     "div_u",  velocity,       alpha);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients(   "solution",     "u",      velocity,       alpha);
  auto &us_dot      = fe_cache.get_values(                "solution_dot", "u_dot",  velocity,       alpha);

  // displacement:
  auto &ds          = fe_cache.get_values(                "solution",     "d",      displacement,   alpha);
  auto &grad_ds     = fe_cache.get_gradients(             "solution",     "grad_d", displacement,   alpha);
  auto &div_ds      = fe_cache.get_divergences(           "solution",     "div_d",  displacement,   alpha);
  auto &Fs          = fe_cache.get_deformation_gradients( "solution",     "Fd",     displacement,   alpha);
  auto &ds_dot      = fe_cache.get_values(                "solution_dot", "d_dot",  displacement,   alpha);

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
          //    displacement:
          auto grad_e = fev[displacement].gradient(i,q);
          //    pressure:
          auto m      = fev[pressure    ].value(i,q);

          // variables:
          //    velocity:
          const Tensor<1, dim, Number>  &u          = us[q];
          const Number                  &div_u      = div_us[q];
          const Tensor<1, dim, Number>  &u_dot      = us_dot[q];
          const Tensor<2, dim, Number>  &grad_u     = grad_us[q];
          const Tensor <2, dim, Number> &sym_grad_u = sym_grad_us[q];
          //    displacement
          const Tensor<1, dim, Number>  &d          = ds[q];
          const Tensor<1, dim, Number>  &d_dot      = ds_dot[q];
          const Tensor<2, dim, Number>  &grad_d     = grad_ds[q];
          const Number                  &div_d      = div_ds[q];
          const Tensor <2, dim, Number> &F          = Fs[q];
          Number                        J           = determinant(F);
          const Tensor <2, dim, Number> &F_inv      = invert(F);
          const Tensor <2, dim, Number> &Ft_inv     = transpose(F_inv);

          //    pressure:
          const Number                  &p          = ps[q];

          // others:
          auto                          J_ale       = J; // jacobian of ALE transformation
          // auto div_u_ale  = (J_ale * (F_inv * u) );
          Tensor <2, dim, Number> Id;
          for (unsigned int i = 0; i<dim; ++i)
            Id[i][i] = p;

          Number my_rho = rho;
          const Tensor <2, dim, Number> sigma       = - Id + my_rho * ( grad_u * F_inv + ( Ft_inv * transpose(grad_u) ) ) ;

          local_residual[i] += (
                                 // time derivative term
                                 rho*inner( u_dot * J_ale , v )
                                 //
                                 + inner( grad_u * ( F_inv * ( u - d_dot ) ) * J_ale , v )
                                 //
                                 + inner( J_ale * sigma * Ft_inv, grad_v)
                                 // stiffness matrix
                                 // + eta*inner(sym_grad_u,grad_v)

                                 // "stiffness" od the displacement
                                 + eta * inner( grad_d , grad_e )
                                 // divergence free constriant
                                 - p * m
                                 // pressure term
                                 - p * div_v
                               )
                               * JxW[q];
        }
    }
}


template <int dim>
void ALENavierStokes<dim>::declare_parameters (ParameterHandler &prm)
{
  NonConservativeInterface<dim,dim,dim+dim+1, ALENavierStokes<dim> >::declare_parameters(prm);
  this->add_parameter(prm, &eta, "eta [Pa s]",    "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &rho, "rho [Kg m^-d]", "1.0", Patterns::Double(0.0));
}

template <int dim>
void ALENavierStokes<dim>::parse_parameters_call_back ()
{
  NonConservativeInterface<dim,dim,dim+dim+1, ALENavierStokes<dim> >::parse_parameters_call_back();
}


template <int dim>
void
ALENavierStokes<dim>::compute_system_operators(const DoFHandler<dim> &dh,
                                               const TrilinosWrappers::BlockSparseMatrix &matrix,
                                               const TrilinosWrappers::BlockSparseMatrix &preconditioner_matrix,
                                               const std::vector<shared_ptr<MAT> >,
                                               LinearOperator<VEC> &system_op,
                                               LinearOperator<VEC> &prec_op) const
{

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector velocity_components(dim);
  DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(velocity_components),
                                    constant_modes);

  P00_preconditioner.reset (new TrilinosWrappers::PreconditionJacobi());
  P11_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
  P22_preconditioner.reset (new TrilinosWrappers::PreconditionJacobi());

  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.constant_modes = constant_modes;
  Amg_data.elliptic = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 0.02;

  P00_preconditioner ->initialize (preconditioner_matrix.block(0,0));
  P11_preconditioner ->initialize (preconditioner_matrix.block(1,1), Amg_data);
  P22_preconditioner ->initialize (preconditioner_matrix.block(2,2));

  // SYSTEM MATRIX:

  std::array<std::array<LinearOperator<TrilinosWrappers::MPI::Vector>, 3>, 3> S;
  for (unsigned int i = 0; i<3; ++i)
    for (unsigned int j = 0; j<3; ++j)
      S[i][j] = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(i,j) );

  // S[1][1] = null_operator(S[1][1]); // d v
  S[0][2] = null_operator(S[0][2]); // d p
  S[2][0] = null_operator(S[2][0]); // p d
  S[2][2] = null_operator(S[2][2]); // p p

  system_op = BlockLinearOperator<TrilinosWrappers::MPI::BlockVector>(S);
  // auto tmp = LinearOperator<TrilinosWrappers::MPI::BlockVector>( S );
  // system_op  = static_cast<LinearOperator<TrilinosWrappers::MPI::BlockVector> &>(tmp);
  std::array<std::array<LinearOperator<TrilinosWrappers::MPI::Vector>, 3>, 3> P;
  for (unsigned int i = 0; i<3; ++i)
    for (unsigned int j = 0; j<3; ++j)
      P[i][j] = linear_operator< TrilinosWrappers::MPI::Vector >( preconditioner_matrix.block(i,j) );
  // PRECONDITIONER MATRIX:

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);

  for (unsigned int i = 0; i<3; ++i)
    for (unsigned int j = 0; j<3; ++j)
      if (i!=j)
        P[i][j] = null_operator< TrilinosWrappers::MPI::Vector >(P[i][j]);

  // P[0][0] = identity_operator< TrilinosWrappers::MPI::Vector >(S[0][0].reinit_domain_vector);
  // P[1][1] = identity_operator< TrilinosWrappers::MPI::Vector >(S[1][1].reinit_domain_vector);
  // P[2][2] = identity_operator< TrilinosWrappers::MPI::Vector >(S[2][2].reinit_domain_vector);
  // for (unsigned int i=0; i<3; ++i)

  P[0][0] = linear_operator<
            dealii::TrilinosWrappers::MPI::Vector,
            dealii::TrilinosWrappers::MPI::Vector,
            TrilinosWrappers::SparseMatrix,
            //dealii::LinearOperator<dealii::TrilinosWrappers::MPI::Vector,
            //                       dealii::TrilinosWrappers::MPI::Vector>,
            TrilinosWrappers::PreconditionJacobi
            >( preconditioner_matrix.block(0,0), *P00_preconditioner  );
  // P[1][1] = linear_operator< >( matrix.block(1,1), *P11_preconditioner  );
  // P[2][2] = linear_operator< >( matrix.block(1,1), *P22_preconditioner  );


  //  P[0][0] = inverse_operator<  >(  S[0][0],
//                                   solver_CG,
//                                   *P00_preconditioner);
  // TrilinosWrappers::MPI::Vector u,v;
  // P[0][0].reinit_domain_vector(u, true);
  // P[0][0].reinit_domain_vector(v, true);
  // std::cout << "vmult... " << std::flush <<std::endl;
  // P[0][0].vmult(u,v);
  P[1][1] = inverse_operator< >(  S[1][1],
                                  solver_CG,
                                  *P11_preconditioner);
  // P[2][2] = S[2][2];
  P[2][2] = inverse_operator< >(  S[2][2],
                                  solver_CG,
                                  *P22_preconditioner);


  // const LinearOperator< typename Solver::vector_type, typename Solver::vector_type > &op, Solver &solver, const Preconditioner &preconditioner)
  prec_op = block_forward_substitution< >(
              BlockLinearOperator<TrilinosWrappers::MPI::BlockVector>(S),
              BlockLinearOperator<TrilinosWrappers::MPI::BlockVector>(P));
  // identity_operator < TrilinosWrappers::MPI::BlockVector >( system_op.reinit_range_vector );
}


template class ALENavierStokes <2>;

#endif
/*! @} */
