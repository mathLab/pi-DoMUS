#ifndef _pidoums_compressible_neo_hookean_h_
#define _pidoums_compressible_neo_hookean_h_

#include "pde_system_interface.h"

#include <deal.II/lac/solver_gmres.h>
#include <deal2lkit/parsed_function.h>

//typedef LATrilinos LAC;

template <int dim, int spacedim, typename LAC=LATrilinos>
class ScalarReactionDiffusionConvection : public PDESystemInterface<dim,spacedim, ScalarReactionDiffusionConvection<dim,spacedim,LAC>, LAC>
{

public:
  ~ScalarReactionDiffusionConvection () {};
  ScalarReactionDiffusionConvection ();

  void declare_parameters (ParameterHandler &prm);

  // interface with the PDESystemInterface :)


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  void compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const;

private:
  double nu;

  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> preconditioner;

  ParsedFunction<dim> convection;

};

template <int dim, int spacedim, typename LAC>
ScalarReactionDiffusionConvection<dim,spacedim, LAC>::
ScalarReactionDiffusionConvection():
  PDESystemInterface<dim,spacedim,ScalarReactionDiffusionConvection<dim,spacedim,LAC>, LAC >("Scalar Reaction Diffusion Convection",
      1,1,
      "FE_Q(2)",
      "u", "1"),
  convection("Convection parameter", dim, "cos(pi*x)*sin(pi*y); -sin(pi*x)*cos(pi*y)")
{}


template <int dim, int spacedim, typename LAC>
void ScalarReactionDiffusionConvection<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, ScalarReactionDiffusionConvection<dim,spacedim,LAC>,LAC >::declare_parameters(prm);
  this->add_parameter(prm, &nu, "Diffusion coefficient", "1.0", Patterns::Double(0.0));
}


template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
ScalarReactionDiffusionConvection<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &residuals,
                       bool ) const
{
  const FEValuesExtractors::Scalar concentration(0);

  ResidualType alpha = 0;
  // Initialize the various solutions, derivatives, etc with the right type for
  // alpha.
  this->reinit (alpha, cell, fe_cache);

  auto &us = fe_cache.get_values("solution", "u", concentration, alpha);
  auto &uts = fe_cache.get_values("solution_dot", "ut", concentration, alpha);
  auto &gradus = fe_cache.get_gradients("solution", "gradu", concentration, alpha);

  const unsigned int n_q_points = us.size();
  auto &JxW = fe_cache.get_JxW_values();

  std::vector<Vector<double> > convection_values(n_q_points, Vector<double>(dim));

  auto &fev = fe_cache.get_current_fe_values();
  convection.vector_value_list(fev.get_quadrature_points(), convection_values);

  for (unsigned int q=0; q<n_q_points; ++q)
    {

      auto &u = us[q];
      auto &ut = uts[q];

      auto &gradu = gradus[q];

      auto &b = convection_values[q];
      for (unsigned int i=0; i<residuals[0].size(); ++i)
        {

          // test functions:
          auto v = fev[concentration].value(i,q);
          auto gradv = fev[concentration].gradient(i,q);

          residuals[0][i] += (ut*v +
                              nu*(gradu*gradv)
                             )*JxW[q];

          for (unsigned int d=0; d<dim; ++d)
            residuals[0][i] += (u*gradu[d]*b[d]*v)*JxW[q];
        }
    }
}



template <int dim, int spacedim, typename LAC>
void
ScalarReactionDiffusionConvection<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
    LinearOperator<LATrilinos::VectorType> &system_op,
    LinearOperator<LATrilinos::VectorType> &prec_op,
    LinearOperator<LATrilinos::VectorType> &) const
{

  preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  preconditioner->initialize(matrices[0]->block(0,0));

  auto P = linear_operator<LATrilinos::VectorType::BlockType>(matrices[0]->block(0,0));

  auto A  = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0) );

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverGMRES<LATrilinos::VectorType::BlockType> solver_GMRES(solver_control_pre);
  auto P_inv     = inverse_operator( P, solver_GMRES, *preconditioner);

  auto P00 = P_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ A }}
    }
  });

  prec_op = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ P00 }} ,
    }
  });
}
#endif
