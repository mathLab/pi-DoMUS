/*! \addtogroup equations
 *  @{
 */

#ifndef _free_swelling_three_fields_h_
#define _free_swelling_three_fields_h_

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

#include "lac/lac_type.h"


typedef LADealII LAC;
// typedef LATrilinos LAC; //


template <int dim, int spacedim>
class FreeSwellingThreeFields : public ConservativeInterface<dim,spacedim,dim+2, FreeSwellingThreeFields<dim,spacedim>, LAC>
{
  typedef FEValuesCache<dim,spacedim> Scratch;
  typedef Assembly::CopyData::piDoMUSPreconditioner<dim,spacedim> CopyPreconditioner;
  typedef Assembly::CopyData::piDoMUSSystem<dim,spacedim> CopySystem;
  typedef BlockSparseMatrix<double> MAT;

public:

  /* specific and useful functions for this very problem */
  ~FreeSwellingThreeFields() {};

  FreeSwellingThreeFields();

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
//
//  void compute_system_operators(const DoFHandler<dim,spacedim> &,
//                                const TrilinosWrappers::BlockSparseMatrix &,
//                                const TrilinosWrappers::BlockSparseMatrix &,
//                                LinearOperator<VEC> &,
//                                LinearOperator<VEC> &) const;

private:
  double T;
  double Omega;
  double G;
  double chi;
  double l0;

  double mu;
  double mu0;
  double l02;
  double l03;
  double l0_3;
  double l0_6;
  const double R=8.314;


//  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> U_prec;
//  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> p_prec;
//  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> c_prec;

};

template <int dim, int spacedim>
FreeSwellingThreeFields<dim,spacedim>::FreeSwellingThreeFields() :
  ConservativeInterface<dim,spacedim,dim+2,FreeSwellingThreeFields<dim,spacedim>, LAC>("Free Swelling Three Fields",
      "FESystem[FE_Q(2)^d-FE_Q(1)-FE_Q(2)]",
      "u,u,u,p,c", "1,1,0;1,0,1;0,1,1", "1,0,0;0,1,0;0,0,1","1,0,0")
{};


template <int dim, int spacedim>
template<typename Number>
void FreeSwellingThreeFields<dim,spacedim>::preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &fe_cache,
    CopyPreconditioner &data,
    Number &energy) const
{
  Number alpha = this->alpha;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);
  auto &us = fe_cache.get_values("solution", "u", displacement, alpha);
  auto &us_dot = fe_cache.get_values("solution_dot", "u", displacement, alpha);

  const FEValuesExtractors::Scalar pressure(dim);
  const FEValuesExtractors::Scalar concentration(dim+1);

  auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);
  auto &cs = fe_cache.get_values("solution", "c", concentration, alpha);

  const unsigned int n_q_points = us.size();

  auto &JxW = fe_cache.get_JxW_values();
  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor <1, dim, Number> &u = us[q];
      const Tensor <1, dim, Number> &u_dot = us_dot[q];
      auto &c = cs[q];
      auto &p = ps[q];

      energy += 0.5*(u*u
                     +p*p
                     +c*c)*JxW[q];
    }
}

template <int dim, int spacedim>
template<typename Number>
void FreeSwellingThreeFields<dim,spacedim>::system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
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

  const FEValuesExtractors::Scalar pressure(dim);
  const FEValuesExtractors::Scalar concentration(dim+1);

  auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);
  auto &cs = fe_cache.get_values("solution", "c", concentration, alpha);

  const unsigned int n_q_points = us.size();

  auto &JxW = fe_cache.get_JxW_values();
  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {

      auto &u = us[q];
      const Tensor <1, dim, Number> &u_dot = us_dot[q];
      const Tensor <2, dim, Number> &F = Fs[q];
      const Tensor<2, dim, Number> C = transpose(F)*F;
      auto &c = cs[q];
      auto &p = ps[q];

      Number I = trace(C);
      Number J = determinant(F);


      Number psi = ( 0.5*G*l0_3*(l02*I - dim)

                     + (l0_3*R*T/Omega)*((Omega*l03*c)*std::log((Omega*l03*c)/(1.+Omega*l03*c))
                                         + chi*((Omega*l03*c)/(1.+Omega*l03*c)) )

                     - (mu0)*c - p*(J-l0_3-Omega*c)) ;

      energy += (u*u_dot + psi)*JxW[q];
    }

}


template <int dim, int spacedim>
void FreeSwellingThreeFields<dim,spacedim>::declare_parameters (ParameterHandler &prm)
{
  ConservativeInterface<dim,spacedim,dim+2, FreeSwellingThreeFields<dim,spacedim>, LAC>::declare_parameters(prm);
  this->add_parameter(prm, &T, "T", "298.0", Patterns::Double(0.0));
  this->add_parameter(prm, &Omega, "Omega", "1e-5", Patterns::Double(0.0));
  this->add_parameter(prm, &chi, "chi", "0.1", Patterns::Double(0.0));
  this->add_parameter(prm, &l0, "l0", "1.5", Patterns::Double(1.0));
  this->add_parameter(prm, &G, "G", "10e3", Patterns::Double(0.0));
  this->add_parameter(prm, &mu, "mu", "-1.8e-22", Patterns::Double(-1000.0));
}

template <int dim, int spacedim>
void FreeSwellingThreeFields<dim,spacedim>::parse_parameters_call_back ()
{
  ConservativeInterface<dim,spacedim,dim+2, FreeSwellingThreeFields<dim,spacedim>, LAC>::parse_parameters_call_back();
  l02 = l0*l0;
  l03 = l02*l0;
  l0_3 = 1./l03;
  l0_6 = 1./(l03*l03);

  mu0 = R*T*(std::log((l03-1.)/l03) + l0_3 + chi*l0_6) + G*Omega/l0;
}
//
//template <int dim, int spacedim>
//void
//FreeSwellingThreeFields<dim,spacedim>::compute_system_operators(const DoFHandler<dim,spacedim> &dh,
//    const TrilinosWrappers::BlockSparseMatrix &matrix,
//    const TrilinosWrappers::BlockSparseMatrix &preconditioner_matrix,
//    LinearOperator<VEC> &system_op,
//    LinearOperator<VEC> &prec_op) const
//{
//
//  std::vector<std::vector<bool> > constant_modes;
//  FEValuesExtractors::Vector velocity_components(0);
//  DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(velocity_components),
//                                    constant_modes);
//
//  p_prec.reset (new TrilinosWrappers::PreconditionJacobi());
//  c_prec.reset (new TrilinosWrappers::PreconditionJacobi());
//  U_prec.reset (new TrilinosWrappers::PreconditionJacobi());
//
////  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
////  Amg_data.constant_modes = constant_modes;
////  Amg_data.elliptic = true;
////  Amg_data.higher_order_elements = true;
////  Amg_data.smoother_sweeps = 2;
////  Amg_data.aggregation_threshold = 0.02;
////
//
//  U_prec->initialize (preconditioner_matrix.block(0,0));
//  p_prec->initialize (preconditioner_matrix.block(1,1));
//  c_prec->initialize (preconditioner_matrix.block(2,2));
//
//
//  // SYSTEM MATRIX:
//  auto A   =   linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,0) );
//  auto Bt  =   linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,1) );
//  auto Z02 = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,2) );
//
//  auto B   =   linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,0) );
//  auto Z11 = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,1) );
//  auto C   =   linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,2) );
//
//  auto Z20 = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(2,0) );
//  auto D   =   linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(2,1) );
//  auto E   =   linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(2,2) );
//
//  auto P0  =  linear_operator< TrilinosWrappers::MPI::Vector >(matrix.block(0,0));
//  auto P1  =  linear_operator< TrilinosWrappers::MPI::Vector >(preconditioner_matrix.block(1,1));
//  auto P2  =  linear_operator< TrilinosWrappers::MPI::Vector >(matrix.block(2,2));
//
//
//  static ReductionControl solver_control_pre(5000, 1e-7);
//  static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
//
//  auto P0_inv = inverse_operator( P0, solver_CG, *U_prec);
//  auto P1_inv = inverse_operator( P1, solver_CG, *p_prec);
//  auto P2_inv = inverse_operator( P2, solver_CG, *c_prec);
//
//
////  auto P0_inv = linear_operator( matrix.block(0,0), *U_prec);
////  auto P1_inv = linear_operator( matrix.block(1,1), *p_prec);
////  auto P2_inv = linear_operator( matrix.block(2,2), *c_prec);
//
//
//  // ASSEMBLE THE PROBLEM:
//  const std::array<std::array<LinearOperator<TrilinosWrappers::MPI::Vector>, 3 >, 3 > matrix_array = {{
//      {{ A,   Bt   , Z02 }},
//      {{ B,   Z11  , C   }},
//      {{ Z20, D    , E   }}
//    }
//  };
//
//  system_op  = block_operator<3, 3, VEC >(matrix_array);
//
//  const std::array<LinearOperator<TrilinosWrappers::MPI::Vector>, 3 > diagonal_array = {{ P0_inv, P1_inv, P2_inv }};
//
//  // system_op = linear_operator<VEC, VEC>(matrix);
//
//  prec_op = block_back_substitution<3, VEC>(matrix_array, diagonal_array);
//
//}


template class FreeSwellingThreeFields <3,3>;

#endif
/*! @} */
