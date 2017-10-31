/*! \addtogroup equations
 *  @{
 */

/**
 *  This interface solves the Eikonal Equation:
 *  \f[
 *     \begin{cases}
 *       \vert nabla u \vert = 1 & \textrm{on }\Omega \\
 *       u = 0 & \textrm{on }\partial\Omega.
 *     \end{cases}
 *  \f]
 *
 *  @note To stabilize this equation we add an extra term:
 *  \f[
 *    \gamma \Delta u.
 *  \f]
 */

#ifndef _pidoums_eikonal_equation_h_
#define _pidoums_eikonal_equation_h_

#include "pde_system_interface.h"
#include <deal2lkit/parsed_preconditioner/amg.h>

template <int dim, int spacedim, typename LAC=LATrilinos>
class EikonalEquation : public PDESystemInterface<dim,spacedim, EikonalEquation<dim,spacedim,LAC>, LAC>
{

public:
  ~EikonalEquation () {};
  EikonalEquation ();

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


  void
  declare_parameters (ParameterHandler &prm);

private:

  /**
   * Multiplicative factor used to stabilize the equation using the laplace.
   */
  double laplacian_stabilization;

  /**
   * Parsed AMG preconditioner used to solve the system.
   */
  mutable ParsedAMGPreconditioner amg;
};

template <int dim, int spacedim, typename LAC>
EikonalEquation<dim,spacedim, LAC>::
EikonalEquation():
  PDESystemInterface<dim,spacedim,EikonalEquation<dim,spacedim,LAC>, LAC >("Eikonal Equation",
      1, 1,
      "FESystem[FE_Q(1)]",
      "d","1"),
  amg("AMG A")
{}

template <int dim, int spacedim, typename LAC>
void EikonalEquation<dim,spacedim,LAC>::
declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, EikonalEquation<dim,spacedim,LAC>,LAC>::
  declare_parameters(prm);

  this->add_parameter(prm, &laplacian_stabilization,
                      "Laplacian stabilization", "1.0",
                      Patterns::Double(0.0),
                      "Laplacian stabilization");
}

template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
EikonalEquation<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> & /*local_energy*/,
                       std::vector<std::vector<ResidualType> > &local_residuals,
                       bool compute_only_system_terms) const
{
  ResidualType alpha = 0;
  double dummy = 0.0;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Scalar distance(0);
  // const FEValuesExtractors::Scalar abs_gradiente;

  auto &d_t_ = fe_cache.get_values("solution_dot", "d", distance, alpha);
  auto &grad_d_ = fe_cache.get_gradients("solution", "grad_d", distance, alpha);
  auto &grad_de_ = fe_cache.get_gradients("explicit_solution", "grad_d", distance, dummy);
  auto &d_ = fe_cache.get_values("solution", "d", distance, alpha);

  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_q_points = d_.size();
  // auto &pcout = this->get_pcout();
  auto &pcout = this->get_pcout();
  auto &fev = fe_cache.get_current_fe_values();
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const auto &grad_d = grad_d_[q];

      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
          auto d_test = fev[distance].value(i,q);
          auto grad_d_test = fev[distance].gradient(i,q);

          local_residuals[0][i] += (
                                     (grad_d*grad_d - 1) * d_test
                                   )*JxW[q];

          // Stabilization:
          auto t = this->get_current_time();
          if ( t < 1.0 )
            local_residuals[0][i] += laplacian_stabilization*(1.0-t)*(
                                       grad_d * grad_d_test
                                     )*JxW[q];
        }
    }

  (void)compute_only_system_terms;
}


template <int dim, int spacedim, typename LAC>
void
EikonalEquation<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
                                                            LinearOperator<LATrilinos::VectorType> &system_op,
                                                            LinearOperator<LATrilinos::VectorType> &prec_op,
                                                            LinearOperator<LATrilinos::VectorType> &) const
{
  typedef LATrilinos::VectorType::BlockType  BVEC;
  typedef LATrilinos::VectorType             VEC;

  const DoFHandler<dim,spacedim> &dh = this->get_dof_handler();
  const ParsedFiniteElement<dim,spacedim> &fe = this->pfe;

  amg.initialize_preconditioner<dim, spacedim>( matrices[0]->block(0,0), fe, dh);

  LinearOperator<BVEC> D  =
    linear_operator<BVEC>( matrices[0]->block(0,0) );

  LinearOperator<BVEC> D_inv =
    linear_operator<BVEC>( matrices[0]->block(0,0), amg);

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<1, 1, VEC>({{
      {{ D }}
    }
  });

  prec_op = block_operator<1, 1, VEC>({{
      {{ D_inv }} ,
    }
  });
}

#endif

/*! @} */
