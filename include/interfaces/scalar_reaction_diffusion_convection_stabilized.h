/*! \addtogroup equations
 *  @{
 */

/**
 *  This interface solves a Scalar Reaction Diffusion Convection Equation:
 *  \f[ TODO:
 *       u_t - \nu * \Delta u + b\cdot\nabla u + cu = f
 *  \f]
 */

#ifndef _pidoums_scalar_reaction_diffusion_convection_h_
#define _pidoums_scalar_reaction_diffusion_convection_h_

#include "pde_system_interface.h"

#include <deal.II/lac/solver_gmres.h>

#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parsed_preconditioner/amg.h>

//typedef LATrilinos LAC;

template <int dim, int spacedim, typename LAC=LATrilinos>
class ScalarReactionDiffusionConvection : public PDESystemInterface<dim,spacedim, ScalarReactionDiffusionConvection<dim,spacedim,LAC>, LAC>
{

public:
  ~ScalarReactionDiffusionConvection () {};
  ScalarReactionDiffusionConvection ();

  void declare_parameters (ParameterHandler &prm);

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

  /**
   * Stabilization parameter
   */
  double alpha;

  /**
   * Reaction parameter
   */
  double gamma;

  /**
   * AMG preconditioner for velocity-velocity matrix.
   */
  mutable ParsedAMGPreconditioner AMG_A;

  /**
   * AMG preconditioner for velocity-velocity matrix.
   */
  mutable ParsedAMGPreconditioner AMG_B;

  ParsedFunction<dim> convection;
};

template <int dim, int spacedim, typename LAC>
ScalarReactionDiffusionConvection<dim,spacedim, LAC>::
ScalarReactionDiffusionConvection():
  PDESystemInterface<dim,spacedim,ScalarReactionDiffusionConvection<dim,spacedim,LAC>, LAC >("Convection Diffusion Reaction Problem",
      1 + dim,
      1,
      "FESystem[FE_Q(1)-FE_Q(2)^d]",
      dim==2 ? "u,q,q" : "u,q,q,q",
      "1,0"),
  AMG_A("AMG for A"),
  AMG_B("AMG for B"),
  convection("Convection parameter", dim, "")
{}


template <int dim, int spacedim, typename LAC>
void ScalarReactionDiffusionConvection<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, ScalarReactionDiffusionConvection<dim,spacedim,LAC >,LAC  >::declare_parameters(prm);

  this->add_parameter(prm,
                      &nu, "Diffusion coefficient",
                      "1.0", Patterns::Double(0.0));

  this->add_parameter(prm,
                      &alpha, "Stabilization coefficient",
                      "1.0", Patterns::Double());

  this->add_parameter(prm,
                      &gamma, "Reaction coefficient",
                      "1.0", Patterns::Double());
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
  const FEValuesExtractors::Vector stabilization_term(1);

  ResidualType alpha = 0;
  // Initialize the various solutions, derivatives, etc with the right type for
  // alpha.
  this->reinit (alpha, cell, fe_cache);

  auto &us = fe_cache.get_values("solution", "u", concentration, alpha);
  auto &uts = fe_cache.get_values("solution_dot", "ut", concentration, alpha);
  auto &gradus = fe_cache.get_gradients("solution", "gradu", concentration, alpha);

  auto &qs = fe_cache.get_values("solution", "q", stabilization_term, alpha);

  const unsigned int n_q_points = us.size();
  auto &JxW = fe_cache.get_JxW_values();

  auto &fev = fe_cache.get_current_fe_values();
  std::vector<Vector<double> > convection_values(n_q_points, Vector<double>(dim));
  convection.vector_value_list(fev.get_quadrature_points(), convection_values);

  for (unsigned int quad=0; quad<n_q_points; ++quad)
    {

      const ResidualType &u = us[quad];
      const ResidualType &ut = uts[quad];

      const Tensor<1, dim, ResidualType> &grad_u = gradus[quad];

      const Tensor<1, dim, ResidualType> &q = qs[quad];

      const Vector<double> &b = convection_values[quad];

      for (unsigned int i=0; i<residuals[0].size(); ++i)
        {

          // test functions:
          auto u_test = fev[concentration].value(i,quad);
          auto grad_u_test = fev[concentration].gradient(i,quad);
          auto q_test = fev[stabilization_term].value(i,quad);

          residuals[0][i] += (
                               ut * u_test +
                               nu*(grad_u*grad_u_test) +
                               gamma * u * u_test
                             )*JxW[quad];

          for (unsigned int h = 0; h<dim; h++)
            residuals[0][i] += (grad_u[h]*b[h])*u_test*JxW[quad];

          residuals[0][i] += (
                               alpha*(grad_u*grad_u_test)
                               -
                               alpha*(q*grad_u_test)
                             )*JxW[quad];

          residuals[0][i] += (
                               alpha*((grad_u - q)*q_test)
                             )*JxW[quad];

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
  typedef LATrilinos::VectorType::BlockType BVEC;
  typedef LATrilinos::VectorType VEC;

  const DoFHandler<dim,spacedim> &dh = this->get_dof_handler();
  const ParsedFiniteElement<dim,spacedim> &fe = this->pfe;

  AMG_A.initialize_preconditioner<dim, spacedim>(
    matrices[0]->block(0,0), fe, dh);
  AMG_B.initialize_preconditioner<dim, spacedim>(
    matrices[0]->block(1,1), fe, dh);

  std::array<std::array<LinearOperator<TrilinosWrappers::MPI::Vector >, 2>, 2> S;
  for (unsigned int i = 0; i<2; ++i)
    for (unsigned int j = 0; j<2; ++j)
      S[i][j] = linear_operator<BVEC>( matrices[0]->block(i,j) );

  std::array<std::array<LinearOperator<TrilinosWrappers::MPI::Vector >, 2>, 2> P;

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverGMRES<LATrilinos::VectorType::BlockType> solver_GMRES(solver_control_pre);

  P[0][0]     = inverse_operator( S[0][0], solver_GMRES, AMG_A);
  P[1][1]     = inverse_operator( S[1][1], solver_GMRES, AMG_B);
  P[1][0]     = null_operator(S[1][0]);
  P[0][1]     = null_operator(S[0][1]);

  // ASSEMBLE THE PROBLEM:
  system_op  = BlockLinearOperator< VEC >(S);
  prec_op = BlockLinearOperator< VEC >(P);
}
#endif

/*! @} */
