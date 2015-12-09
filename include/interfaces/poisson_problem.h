#ifndef _pidoums_poisson_h_
#define _pidoums_poisson_h_

#include "interface.h"

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/solver_cg.h>

//typedef LATrilinos LAC;

template <int dim, int spacedim, typename LAC=LATrilinos>
class PoissonProblem : public Interface<dim,spacedim, PoissonProblem<dim,spacedim,LAC>, LAC>
{

public:
  ~PoissonProblem () {};
  PoissonProblem ();

  // interface with the Interface :)


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                const std::vector<shared_ptr<LATrilinos::BlockMatrix> >,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const;

private:
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> preconditioner;


};

template <int dim, int spacedim, typename LAC>
PoissonProblem<dim,spacedim, LAC>::
PoissonProblem():
  Interface<dim,spacedim,PoissonProblem<dim,spacedim,LAC>, LAC >("Poisson problem",
      1,1,
      "FESystem[FE_Q(1)]",
      "u","1")
{
  this->init();
}



template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
PoissonProblem<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &local_residuals,
                       bool compute_only_system_terms) const
{
  const FEValuesExtractors::Scalar s(0);

  ResidualType rt = this->alpha; // dummy number to define the type of variables
  this->reinit (rt, cell, fe_cache);
  auto &uts = fe_cache.get_values("solution_dot", "u", s, rt);
  auto &gradus = fe_cache.get_gradients("solution", "u", s, rt);

  const unsigned int n_q_points = uts.size();
  auto &JxW = fe_cache.get_JxW_values();

  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      auto &ut = uts[q];
      auto &gradu = gradus[q];
      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
          auto v = fev[s].value(i,q);
          auto gradv = fev[s].gradient(i,q);
          local_residuals[0][i] += (
                                     ut*v
                                     +
                                     gradu*gradv
                                   )*JxW[q];
        }

      (void)compute_only_system_terms;

    }


}


template <int dim, int spacedim, typename LAC>
void
PoissonProblem<dim,spacedim,LAC>::compute_system_operators(const DoFHandler<dim,spacedim> &,
                                                           const std::vector<shared_ptr<LATrilinos::BlockMatrix> > matrices,
                                                           LinearOperator<LATrilinos::VectorType> &system_op,
                                                           LinearOperator<LATrilinos::VectorType> &prec_op) const
{

  preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  preconditioner->initialize(matrices[1]->block(0,0));
  auto P = linear_operator<LATrilinos::VectorType::BlockType>(matrices[0]->block(0,0));

  auto A  = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0) );

  static ReductionControl solver_control_pre(5000, 1e-4);
  static SolverCG<LATrilinos::VectorType::BlockType> solver_CG(solver_control_pre);
  auto P_inv     = inverse_operator( P, solver_CG, *preconditioner);

  auto P00 = P_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ A }}
    }
  });

  prec_op = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ P00}} ,
    }
  });
}


template class PoissonProblem <2,2, LADealII>;
template class PoissonProblem <2,3, LADealII>;
template class PoissonProblem <3,3, LADealII>;


template class PoissonProblem <2,2, LATrilinos>;
template class PoissonProblem <2,3, LATrilinos>;
template class PoissonProblem <3,3, LATrilinos>;





#endif
