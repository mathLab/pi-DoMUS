#ifndef _compressible_neo_hookean_h_
#define _compressible_neo_hookean_h_

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


template <int dim, int spacedim>
class CompressibleNeoHookeanInterface : public ConservativeInterface<dim,spacedim,dim, CompressibleNeoHookeanInterface<dim,spacedim>, LADealII >
{
  typedef FEValuesCache<dim,spacedim> Scratch;
  typedef Assembly::CopyData::piDoMUSPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::piDoMUSSystem<dim,dim> CopySystem;
  typedef BlockVector<double> VEC;

public:

  /* specific and useful functions for this very problem */
  ~CompressibleNeoHookeanInterface() {};

  CompressibleNeoHookeanInterface();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();


  /* these functions MUST have the follwowing names
   *  because they are called by the ConservativeInterface class
   */

  template<typename Number>
  void preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                             Scratch &,
                             CopyPreconditioner &,
                             Number &energy) const;

  template<typename Number>
  void system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                     Scratch &,
                     CopySystem &,
                     Number &energy) const;

  void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                const TrilinosWrappers::BlockSparseMatrix &,
                                const TrilinosWrappers::BlockSparseMatrix &,
                                LinearOperator<VEC> &,
                                LinearOperator<VEC> &) const;

private:
  double E;
  double nu;
  double mu;
  double lambda;


  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

};

template <int dim, int spacedim>
CompressibleNeoHookeanInterface<dim,spacedim>::CompressibleNeoHookeanInterface() :
  ConservativeInterface<dim,spacedim,dim,CompressibleNeoHookeanInterface<dim,spacedim>,LADealII >("Compressible NeoHookean Interface",
      "FESystem[FE_Q(1)^d]",
      "u,u,u", "1", "1","1")
{};


template <int dim, int spacedim>
template<typename Number>
void CompressibleNeoHookeanInterface<dim,spacedim>::preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &fe_cache,
    CopyPreconditioner &data,
    Number &energy) const
{
  Number alpha = this->alpha;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);
  auto &us = fe_cache.get_values("solution", "u", displacement, alpha);

  const unsigned int n_q_points = us.size();

  auto &JxW = fe_cache.get_JxW_values();
  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor <1, dim, Number> &u = us[q];

      energy += 0.5*(u*u)*JxW[q];
    }
}

template <int dim, int spacedim>
template<typename Number>
void CompressibleNeoHookeanInterface<dim,spacedim>::system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &fe_cache,
    CopySystem &data,
    Number &energy) const
{
  Number alpha = this->alpha;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);
  auto &us = fe_cache.get_values("solution", "u", displacement, alpha);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", displacement, alpha);
  auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, alpha);

  const unsigned int n_q_points = us.size();

  auto &JxW = fe_cache.get_JxW_values();
  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {

      const Tensor <1, dim, Number> &u = us[q];
      const Tensor <1, dim, Number> &u_dot = us_dot[q];
      const Tensor <2, dim, Number> &F = Fs[q];
      const Tensor<2, dim, Number> C = transpose(F)*F;

      Number Ic = trace(C);
      Number J = determinant(F);
      Number lnJ = std::log (J);

      Number psi = (mu/2.)*(Ic-dim) - mu*lnJ + (lambda/2.)*(lnJ)*(lnJ);

      energy += (psi)*JxW[q];

    }

}


template <int dim, int spacedim>
void CompressibleNeoHookeanInterface<dim,spacedim>::declare_parameters (ParameterHandler &prm)
{
  ConservativeInterface<dim,spacedim,dim, CompressibleNeoHookeanInterface<dim,spacedim>, LADealII >::declare_parameters(prm);
  this->add_parameter(prm, &E, "Young's modulus", "10.0", Patterns::Double(0.0));
  this->add_parameter(prm, &nu, "Poisson's ratio", "0.3", Patterns::Double(0.0));
}

template <int dim, int spacedim>
void CompressibleNeoHookeanInterface<dim,spacedim>::parse_parameters_call_back ()
{
  ConservativeInterface<dim,spacedim,dim, CompressibleNeoHookeanInterface<dim,spacedim>, LADealII >::parse_parameters_call_back();
  mu = E/(2.0*(1.+nu));
  lambda = (E *nu)/((1.+nu)*(1.-2.*nu));
}

// template <int dim, int spacedim>
// void
// CompressibleNeoHookeanInterface<dim,spacedim>::compute_system_operators(const DoFHandler<dim,spacedim> &dh,
//     const TrilinosWrappers::BlockSparseMatrix &matrix,
//     const TrilinosWrappers::BlockSparseMatrix &preconditioner_matrix,
//     LinearOperator<VEC> &system_op,
//     LinearOperator<VEC> &prec_op) const
// {
//
//   std::vector<std::vector<bool> > constant_modes;
//   FEValuesExtractors::Vector displacement(0);
//   DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(displacement),
//                                     constant_modes);
//
//   Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
//   Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
//
//   TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
//   Amg_data.constant_modes = constant_modes;
//   Amg_data.elliptic = true;
//   Amg_data.higher_order_elements = true;
//   Amg_data.smoother_sweeps = 2;
//   Amg_data.aggregation_threshold = 0.02;
//
// //  Mp_preconditioner->initialize (preconditioner_matrix.block(0,0));
//   Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
//                                   Amg_data);
//
//
//   auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,0) );
//
//   static ReductionControl solver_control_pre(5000, 1e-8);
//   static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
//   auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);
//
//   auto P00 = A_inv;
//
//   // ASSEMBLE THE PROBLEM:
//   system_op  = block_operator<1, 1, VEC >({{
//       {{ A }}
//     }
//   });
//
//
//   //const auto S = linear_operator<VEC>(matrix);
//
//   prec_op = block_operator<1, 1, VEC >({{
//       {{ P00}} ,
//     }
//   });
// }


template class CompressibleNeoHookeanInterface <3,3>;

#endif
