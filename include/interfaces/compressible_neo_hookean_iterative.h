#ifndef _pidoums_compressible_neo_hookean_h_
#define _pidoums_compressible_neo_hookean_h_

#include "interface.h"

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/solver_cg.h>


template <int dim, int spacedim>
class CompressibleNeoHookeanInterface : public Interface<dim,spacedim,dim, CompressibleNeoHookeanInterface<dim,spacedim> >
{
  typedef TrilinosWrappers::MPI::BlockVector BVEC;
  typedef TrilinosWrappers::MPI::Vector VEC;
  typedef TrilinosWrappers::BlockSparseMatrix MAT;

public:
  ~CompressibleNeoHookeanInterface () {};
  CompressibleNeoHookeanInterface ();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  // interface with the Interface :)

  unsigned int get_number_of_matrices() const
  {
    return 2;
  }

  void set_matrices_coupling (std::vector<std::string> &couplings) const
  {
    couplings[0] = "1";
    couplings[1] = "1";
  };


  template <typename EnergyType, typename ResidualType>
  void set_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                  FEValuesCache<dim,spacedim> &scratch,
                                  std::vector<EnergyType> &energies,
                                  std::vector<std::vector<ResidualType> > &local_residuals,
                                  bool compute_only_system_matrix) const;


  void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                const std::vector<shared_ptr<MAT> >,
                                LinearOperator<BVEC> &,
                                LinearOperator<BVEC> &) const;

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
  Interface<dim,spacedim,dim,CompressibleNeoHookeanInterface<dim,spacedim> >("Compressible NeoHookean Interface",
      "FESystem[FE_Q(1)^d]",
      "u,u,u","1")
{}


template <int dim, int spacedim>
void CompressibleNeoHookeanInterface<dim,spacedim>::declare_parameters (ParameterHandler &prm)
{
  Interface<dim,spacedim,dim, CompressibleNeoHookeanInterface<dim,spacedim> >::declare_parameters(prm);
  this->add_parameter(prm, &E, "Young's modulus", "10.0", Patterns::Double(0.0));
  this->add_parameter(prm, &nu, "Poisson's ratio", "0.3", Patterns::Double(0.0));
}

template <int dim, int spacedim>
void CompressibleNeoHookeanInterface<dim,spacedim>::parse_parameters_call_back ()
{
  Interface<dim,spacedim,dim, CompressibleNeoHookeanInterface<dim,spacedim> >::parse_parameters_call_back();
  mu = E/(2.0*(1.+nu));
  lambda = (E *nu)/((1.+nu)*(1.-2.*nu));
}



template <int dim, int spacedim>
template <typename EnergyType, typename ResidualType>
void
CompressibleNeoHookeanInterface<dim,spacedim>::
set_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                           FEValuesCache<dim,spacedim> &fe_cache,
                           std::vector<EnergyType> &energies,
                           std::vector<std::vector<ResidualType> > &,
                           bool compute_only_system_matrix) const
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

      auto psi = (mu/2.)*(Ic-dim) - mu*lnJ + (lambda/2.)*(lnJ)*(lnJ);

      energies[0] += (psi)*JxW[q];

      if (!compute_only_system_matrix)
        energies[1] += 0.5*u*u*JxW[q];

      (void)compute_only_system_matrix;

    }


}

template <int dim, int spacedim>
void
CompressibleNeoHookeanInterface<dim,spacedim>::compute_system_operators(const DoFHandler<dim,spacedim> &,
    const std::vector<shared_ptr<MAT> > matrices,
    LinearOperator<BVEC> &system_op,
    LinearOperator<BVEC> &prec_op) const
{

  preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  preconditioner->initialize(matrices[1]->block(0,0));
  auto P = linear_operator<VEC>(matrices[1]->block(0,0));

  auto A  = linear_operator<VEC>( matrices[0]->block(0,0) );

  static ReductionControl solver_control_pre(5000, 1e-4);
  static SolverCG<VEC> solver_CG(solver_control_pre);
  auto P_inv     = inverse_operator( P, solver_CG, *preconditioner);

  auto P00 = P_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<1, 1, BVEC >({{
      {{ A }}
    }
  });


  //const auto S = linear_operator<VEC>(matrix);

  prec_op = block_operator<1, 1, BVEC >({{
      {{ P00}} ,
    }
  });
}


template class CompressibleNeoHookeanInterface <3,3>;




#endif
