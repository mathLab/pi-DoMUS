#ifndef _pidoums_compressible_neo_hookean_h_
#define _pidoums_compressible_neo_hookean_h_

#include "interface.h"

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/solver_cg.h>

typedef LATrilinos LAC;

template <int dim, int spacedim>
class CompressibleNeoHookeanInterface : public Interface<dim,spacedim, CompressibleNeoHookeanInterface<dim,spacedim>, LAC>
{

public:
  ~CompressibleNeoHookeanInterface () {};
  CompressibleNeoHookeanInterface ();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  // interface with the Interface :)


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                const std::vector<shared_ptr<LAC::BlockMatrix> >,
                                LinearOperator<LAC::VectorType> &,
                                LinearOperator<LAC::VectorType> &) const;

private:
  double E;
  double nu;
  double mu;
  double lambda;


  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> preconditioner;


};

template <int dim, int spacedim>
CompressibleNeoHookeanInterface<dim,spacedim>::
CompressibleNeoHookeanInterface():
  Interface<dim,spacedim,CompressibleNeoHookeanInterface<dim,spacedim> >(dim,2,
      "Compressible NeoHookean Interface",
      "FESystem[FE_Q(1)^d]",
      "u,u,u","1")
{}


template <int dim, int spacedim>
void CompressibleNeoHookeanInterface<dim,spacedim>::declare_parameters (ParameterHandler &prm)
{
  Interface<dim,spacedim, CompressibleNeoHookeanInterface<dim,spacedim> >::declare_parameters(prm);
  this->add_parameter(prm, &E, "Young's modulus", "10.0", Patterns::Double(0.0));
  this->add_parameter(prm, &nu, "Poisson's ratio", "0.3", Patterns::Double(0.0));
}

template <int dim, int spacedim>
void CompressibleNeoHookeanInterface<dim,spacedim>::parse_parameters_call_back ()
{
  mu = E/(2.0*(1.+nu));
  lambda = (E *nu)/((1.+nu)*(1.-2.*nu));
}



template <int dim, int spacedim>
template <typename EnergyType, typename ResidualType>
void
CompressibleNeoHookeanInterface<dim,spacedim>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &energies,
                       std::vector<std::vector<ResidualType> > &,
                       bool compute_only_system_terms) const
{
  const FEValuesExtractors::Vector displacement(0);

  ////////// conservative section
  //
  EnergyType et = 0; // dummy number to define the type of variables
  this->reinit (et, cell, fe_cache);
  auto &us = fe_cache.get_values("solution", "u", displacement, et);
  auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, et);

  const unsigned int n_q_points = us.size();
  auto &JxW = fe_cache.get_JxW_values();


  for (unsigned int q=0; q<n_q_points; ++q)
    {
      ///////////////////////// energetic contribution
      auto &u = us[q];
      auto &F = Fs[q];
      auto C = transpose(F)*F;

      auto Ic = trace(C);
      auto J = determinant(F);
      auto lnJ = std::log (J);

      EnergyType psi = (mu/2.)*(Ic-dim) - mu*lnJ + (lambda/2.)*(lnJ)*(lnJ);

      energies[0] += (psi)*JxW[q];

      if (!compute_only_system_terms)
        energies[1] += 0.5*u*u*JxW[q];

      (void)compute_only_system_terms;

    }


}

template <int dim, int spacedim>
void
CompressibleNeoHookeanInterface<dim,spacedim>::compute_system_operators(const DoFHandler<dim,spacedim> &,
    const std::vector<shared_ptr<LAC::BlockMatrix> > matrices,
    LinearOperator<LAC::VectorType> &system_op,
    LinearOperator<LAC::VectorType> &prec_op) const
{

  preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  preconditioner->initialize(matrices[1]->block(0,0));
  auto P = linear_operator<LAC::VectorType::BlockType>(matrices[1]->block(0,0));

  auto A  = linear_operator<LAC::VectorType::BlockType>( matrices[0]->block(0,0) );

  static ReductionControl solver_control_pre(5000, 1e-4);
  static SolverCG<LAC::VectorType::BlockType> solver_CG(solver_control_pre);
  auto P_inv     = inverse_operator( P, solver_CG, *preconditioner);

  auto P00 = P_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<1, 1, LAC::VectorType>({{
      {{ A }}
    }
  });

  prec_op = block_operator<1, 1, LAC::VectorType>({{
      {{ P00}} ,
    }
  });
}


template class CompressibleNeoHookeanInterface <3,3>;




#endif
