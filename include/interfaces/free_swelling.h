#ifndef _compressible_neo_hookean_h_
#define _compressible_neo_hookean_h_

#include "conservative_interface.h"
#include "parsed_function.h"


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
class FreeSwellingThreeField : public ConservativeInterface<dim,spacedim,dim+2, FreeSwellingThreeField<dim,spacedim> >
{
  typedef FEValuesCache<dim,spacedim> Scratch;
  typedef Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> CopyPreconditioner;
  typedef Assembly::CopyData::NFieldsSystem<dim,spcedim> CopySystem;
  typedef TrilinosWrappers::MPI::BlockVector VEC;

public:

  /* specific and useful functions for this very problem */
  ~FreeSwellingThreeField() {};

  FreeSwellingThreeField();

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


  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

};

template <int dim, int spacedim>
FreeSwellingThreeField<dim,spacedim>::FreeSwellingThreeField() :
  ConservativeInterface<dim,spacedim,dim+2,FreeSwellingThreeField<dim,spacedim> >("Free Swelling Three Fields",
      "FESystem[FE_Q(2)^d-FE_Q(2)-FE_Q(1)]",
      "u,u,u,c,p", "1,0,1;0,1,1;1,1,0", "1,0,0;0,1,0;0,0,1","1,0,0")
{};


template <int dim, int spacedim>
template<typename Number>
void FreeSwellingThreeField<dim,spacedim>::preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &fe_cache,
    CopyPreconditioner &data,
    Number &energy) const
{
  Number alpha = this->alpha;

  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
  fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);

  const FEValuesExtractors::Vector displacement(0);
  auto &us = fe_cache.get_values("solution", "u", displacement, alpha);
  auto &us_dot = fe_cache.get_values("solution_dot", "u", displacement, alpha);

	const FEValuesExtractors::Scalar concentration(dim);
	const FEValuesExtractors::Scalar pressure(dim+1);

	auto &cs = fe_cache.get_values("solution", "c", concentration, alpha);
	auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);

  const unsigned int n_q_points = us.size();

  auto &JxW = fe_cache.get_JxW_values();
  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor <1, dim, Number> &u = us[q];
			auto &c = cs[q];
			auto &p = ps[q];

      energy += 0.5*(u*u_dot
					          +p*p
										+c*c)*JxW[q];
    }
}

template <int dim, int spacedim>
template<typename Number>
void FreeSwellingThreeField<dim,spacedim>::system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &fe_cache,
    CopySystem &data,
    Number &energy) const
{
  Number alpha = this->alpha;

  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
  fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);
  this->fix_solution_dot_derivative(fe_cache, alpha);

  const FEValuesExtractors::Vector displacement(0);
  auto &us = fe_cache.get_values("solution", "u", displacement, alpha);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", displacement, alpha);
  auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, alpha);

	const FEValuesExtractors::Scalar concentration(dim);
	const FEValuesExtractors::Scalar pressure(dim+1);

	auto &cs = fe_cache.get_values("solution", "c", concentration, alpha);
	auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);

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


//			Number psi = 0.5*G*l0_3*(l02*I - dim) + (l0_3*R*T/Omega)*((l03*J-1.)*std::log(1.-1./(l03*J)) + chi*(1.-1./(l03*J)) ) - mu*(l03*J -1.)/(l03*Omega);
			Number psi = 0.5*G*l0_3*(l02*I - dim) + (l0_3*R*T/Omega)*((Omega*c)*std::log((Omega*c)/(1.+Omega*c)) + chi*((Omega*c)/(1.+Omega*c)) ) - mu*c*l0_3 - p*(J-l0_3-Omega*c);

      energy += (u*u_dot + psi)*JxW[q];
    }

}


template <int dim, int spacedim>
void FreeSwellingThreeField<dim,spacedim>::declare_parameters (ParameterHandler &prm)
{
  ConservativeInterface<dim,spacedim,dim+2, FreeSwellingThreeField<dim,spacedim> >::declare_parameters(prm);
  this->add_parameter(prm, &T, "T", "298.0", Patterns::Double(0.0));
  this->add_parameter(prm, &Omega, "Omega", "1e-5", Patterns::Double(0.0));
  this->add_parameter(prm, &chi, "chi", "0.1", Patterns::Double(0.0));
  this->add_parameter(prm, &l0, "l0", "1.5", Patterns::Double(1.0));
  this->add_parameter(prm, &G, "G", "10e3", Patterns::Double(0.0));
  this->add_parameter(prm, &mu, "mu", "-1.8e-22", Patterns::Double(-1000.0));
}

template <int dim, int spacedim>
void FreeSwellingThreeField<dim,spacedim>::parse_parameters_call_back ()
{
  ConservativeInterface<dim,spacedim,dim+2, FreeSwellingThreeField<dim,spacedim> >::parse_parameters_call_back();
  l02 = l0*l0;
  l03 = l02*l0;
  l0_3 = 1./l03;
  l0_6 = 1./(l03*l03);

  mu0 = R*T*(std::log((l03-1.)/l03) + l0_3 + chi*l0_6) + G*Omega/l0;
}

template <int dim, int spacedim>
void
FreeSwellingThreeField<dim,spacedim>::compute_system_operators(const DoFHandler<dim,spacedim> &dh,
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
  Mf_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.constant_modes = constant_modes;
  Amg_data.elliptic = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 0.02;

  Mc_preconditioner->initialize (preconditioner_matrix.block(1,1));
  Mp_preconditioner->initialize (preconditioner_matrix.block(2,2));
  Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
                                  Amg_data);


  // SYSTEM MATRIX:
  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,0) );
  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,1) );
  //  auto B =  transpose_operator(Bt);
  auto B     = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,0) );
  auto C     = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(2,0) );
  auto D     = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(2,2) );
  auto Z02 = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,2) );
  auto Z11 = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,1) );
  auto Z12 = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,2) );
  auto Z21 = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(2,1) );
	
  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<3, 3, VEC >({{
      {{ A, Bt   , Z02 }},
      {{ B, Z11  , Z12 }},
			{{ C, Z21  , D   }}
    }
  });



  auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >( preconditioner_matrix.block(1,1) );
  auto Mf    = linear_operator< TrilinosWrappers::MPI::Vector >( preconditioner_matrix.block(2,2) );

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
  auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);
  auto Schur_inv = inverse_operator( Mp, solver_CG, *Mp_preconditioner);
	auto Phi_inv   = inverse_operator( Mf,solver_CG, *Mf_preconditioner);

  auto P00 = A_inv;
  auto P01 = null_operator(Bt);
  auto P10 = Schur_inv * B * A_inv;
  auto P11 = -1 * Schur_inv;
  auto Z20 = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(2,0) );


  //const auto S = linear_operator<VEC>(matrix);

  prec_op = block_operator<3, 3, VEC >({{
      {{ P00, P01, Z02 }} ,
      {{ P10, P11, Z12 }},
			{{ Z20, Z21, Phi_inv}}
    }
  });
}


template class FreeSwellingThreeField <3,3>;

#endif
