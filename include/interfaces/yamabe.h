#ifndef _yamabe_h_
#define _yamabe_h_

#include "pde_system_interface.h"


template <int dim, int spacedim, typename LAC=LATrilinos>
class YamabeProblem : public PDESystemInterface<dim,spacedim, YamabeProblem<dim,spacedim,LAC>, LAC>
{

public:
  ~YamabeProblem () {}
  YamabeProblem ();

  // interface with the PDESystemInterface :)


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  void compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> >,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const;

private:
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> preconditioner;

};

template <int dim, int spacedim, typename LAC>
YamabeProblem<dim,spacedim, LAC>::
YamabeProblem():
  PDESystemInterface<dim,spacedim,YamabeProblem<dim,spacedim,LAC>, LAC >("Yamabe equation",
      1, // numero di componenti
      1, // numero di matrici che assemblo
      "FESystem[FE_Q(1)]", // tipo di elemento finito
      "u", // nome della variabile
      "0") // 0 se variabile algebrica. 1 se differenziale (du/dt)
{}



template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
YamabeProblem<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &local_residuals,
                       bool compute_only_system_terms) const
{

  const FEValuesExtractors::Scalar s(0);

  ResidualType rt = 0; // dummy number to define the type of variables
  double dd=0;
  this->reinit (dd, cell, fe_cache);
  this->reinit (rt, cell, fe_cache);
  auto &gradus = fe_cache.get_gradients("solution", "u", s, rt);
//  auto &us = fe_cache.get_values("explicit_solution", "u", s, dd);
  auto &us = fe_cache.get_values("solution", "u", s, rt);

  const unsigned int n_q_points = us.size();
  auto &JxW = fe_cache.get_JxW_values();

  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      auto &u = us[q];
      auto &gradu = gradus[q];
      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
          auto v = fev[s].value(i,q);
          auto gradv = fev[s].gradient(i,q);
          local_residuals[0][i] += (
                                     (3./4.)*v*(u*u*u*u*u)
                                     +
                                     gradu*gradv
                                   )*JxW[q];
        }

      (void)compute_only_system_terms;

    }

}


template <int dim, int spacedim, typename LAC>
void
YamabeProblem<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > matrices,
                                                          LinearOperator<LATrilinos::VectorType> &system_op,
                                                          LinearOperator<LATrilinos::VectorType> &prec_op,
                                                          LinearOperator<LATrilinos::VectorType> &) const
{

  preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  preconditioner->initialize(matrices[0]->block(0,0));

  auto A  = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0) );

  LinearOperator<LATrilinos::VectorType::BlockType> P_inv;

  P_inv = linear_operator<LATrilinos::VectorType::BlockType>(matrices[0]->block(0,0), *preconditioner);

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

#endif
