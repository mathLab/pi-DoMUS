#ifndef _pidoums_poisson_h_
#define _pidoums_poisson_h_

#include "pde_system_interface.h"

using namespace dealii;

template <int dim, int spacedim, typename LAC=LATrilinos>
class PoissonProblem : public PDESystemInterface<dim,spacedim, PoissonProblem<dim,spacedim,LAC>, LAC>
{

public:
  ~PoissonProblem () {};
  PoissonProblem ();

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

  virtual void connect_to_signals() const
  {
    // first of all we get the struct Signals from pidomus
    auto &signals = this->get_signals();

    // we can connect calling .connect( and defining a lambda
    signals.fix_initial_conditions.connect(
      [this](typename LAC::VectorType &, typename LAC::VectorType &)
    {
      std::cout << "ciao mondo" << std::endl;
    }
    );

    // or we can define a lambda first
    auto l =  [this](typename LAC::VectorType &, typename LAC::VectorType &)
    {
      std::cout << "ho raffinato" << std::endl;
    };

    // and then attach the just defined lambda
    signals.fix_solutions_after_refinement.connect(l);


    // herebelow, we connect to all the begin_* signals available in piDoMUS
    auto &pcout = this->get_pcout();
    signals.begin_make_grid_fe.connect(
      [&]()
    {
      pcout << "#########  make_grid_fe"<<std::endl;
    });
    signals.begin_setup_dofs.connect(
      [&]()
    {
      pcout << "#########  setup_dofs"<<std::endl;
    });
    signals.begin_refine_mesh.connect(
      [&]()
    {
      pcout << "#########  refine_mesh"<<std::endl;
    });
    signals.begin_setup_jacobian.connect(
      [&]()
    {
      pcout << "#########  setup_jacobian"<<std::endl;
    });
    signals.begin_residual.connect(
      [&]()
    {
      pcout << "#########  residual"<<std::endl;
    });
    signals.begin_solve_jacobian_system.connect(
      [&]()
    {
      pcout << "#########  solve_jacobian_system"<<std::endl;
    });
    signals.begin_refine_and_transfer_solutions.connect(
      [&]()
    {
      pcout << "#########  refine_and_transfer_solutions"<<std::endl;
    });
    signals.begin_assemble_matrices.connect(
      [&]()
    {
      pcout << "#########  assemble_matrices"<<std::endl;
    });
    signals.begin_solver_should_restart.connect(
      [&]()
    {
      pcout << "#########  solver_should_restart"<<std::endl;
    });

  }
private:
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> preconditioner;
};

template <int dim, int spacedim, typename LAC>
PoissonProblem<dim,spacedim, LAC>::
PoissonProblem():
  PDESystemInterface<dim,spacedim,PoissonProblem<dim,spacedim,LAC>, LAC >("Poisson problem",
      1,1,
      "FESystem[FE_Q(1)]",
      "u","1")
{}



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

  ResidualType rt = 0; // dummy number to define the type of variables
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
PoissonProblem<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > matrices,
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
