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

#ifndef _pidoums_eikonal_equation_two_steps_h_
#define _pidoums_eikonal_equation_two_steps_h_

#include "pde_system_interface.h"

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/base/work_stream.h>

#include <deal2lkit/parsed_preconditioner/amg.h>
#include <deal2lkit/sacado_tools.h>

#include <Sacado_Fad_SimpleFadOps.hpp>

namespace EikonalUtilities
{
////////////////////////////////////////////////////////////////////////////////
/// Structs and classes:

  template <int dim>
  struct CopyNorm
  {
    CopyNorm ()
    {};

    ~CopyNorm ()
    {};

    CopyNorm (const CopyNorm &data)
      :
      local_norm(data.local_norm)
    {};

    double local_norm;
  };

}

template <int dim, int spacedim, typename LAC=LATrilinos>
class EikonalEquation : public PDESystemInterface<dim,spacedim, EikonalEquation<dim,spacedim,LAC>, LAC>
{

public:
  ~EikonalEquation () {}
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

  void
  solution_preprocessing(FEValuesCache<dim,spacedim> &fe_cache) const;

  void
  assemble_local_norm(
    const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    FEValuesCache<dim,spacedim> &fe_cache,
    EikonalUtilities::CopyNorm<dim> &data
  )const;

private:

  /**
   * Threshold.for the heat equation
   */
  // double heat_threshold;
  bool use_heat_equation;
  double auxiliary_function_stabilization;
  mutable double grad_a_norm;


  /**
   * Parsed AMG preconditioner used to solve the system.
   */
  mutable ParsedAMGPreconditioner AMG_A;

  /**
   * Parsed AMG preconditioner used to solve the system.
   */
  mutable ParsedAMGPreconditioner AMG_B;
};

template <int dim, int spacedim, typename LAC>
EikonalEquation<dim,spacedim, LAC>::
EikonalEquation():
  PDESystemInterface<dim,spacedim,EikonalEquation<dim,spacedim,LAC>, LAC >("Eikonal Equation",
      2, 1,
      "FESystem[FE_Q(1)-FE_Q(1)]",
      "d,a","1,1"),
  AMG_A("AMG A"),
  AMG_B("AMG B")
{}

template <int dim, int spacedim, typename LAC>
void EikonalEquation<dim,spacedim,LAC>::
declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, EikonalEquation<dim,spacedim,LAC>,LAC>::
  declare_parameters(prm);

  // this->add_parameter(prm, &heat_threshold,
  //                     "Heat threshold", "2e-2",
  //                     Patterns::Double(0.0),
  //                     "Heat threshold");
  this->add_parameter(prm, &auxiliary_function_stabilization,
                      "Laplacian stabilization", "1e-3",
                      Patterns::Double(0.0),
                      "Laplacian stabilization");
  this->add_parameter(prm, &use_heat_equation,
                      "Use Heat Equation", "true",
                      Patterns::Bool(),
                      "Use Heat Equation as first step");

}

template <int dim, int spacedim, typename LAC>
void EikonalEquation<dim,spacedim,LAC>::
solution_preprocessing(FEValuesCache<dim,spacedim> &fe_cache) const
{
  double global_norm = 0;

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;

  auto local_copy = [this, &global_norm]
                    (const EikonalUtilities::CopyNorm<dim> &data)
  {
    global_norm += data.local_norm;
  };

  auto local_assemble = [this]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         EikonalUtilities::CopyNorm<dim> &data)
  {
    assemble_local_norm(cell, scratch, data);
  };


  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   this->get_dof_handler().begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   this->get_dof_handler().end()),
       local_assemble,
       local_copy,
       fe_cache,
       EikonalUtilities::CopyNorm<dim>());

  auto &cache = fe_cache.get_cache();

  global_norm = Utilities::MPI::sum(global_norm, MPI_COMM_WORLD);

  cache.template add_copy<double >(global_norm, "global_norm");
  // auto &pcout = this->get_pcout();
  // pcout << "Grad norm = " << grad_a_norm << std::endl;
  grad_a_norm = std::sqrt(global_norm);
}

template <int dim, int spacedim, typename LAC>
void
EikonalEquation<dim,spacedim, LAC>::
assemble_local_norm(
  const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
  FEValuesCache<dim,spacedim> &fe_cache,
  EikonalUtilities::CopyNorm<dim> &data
)const
{
  double alpha = 0;

  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Scalar auxiliary_function(1);
  const FEValuesExtractors::Scalar distance(0);

  auto &a_ = fe_cache.get_values("solution", "a", auxiliary_function, alpha);
  auto &grad_a_ = fe_cache.get_gradients("solution", "grad_a", auxiliary_function, alpha);

  auto &JxW = fe_cache.get_JxW_values();
  auto &q_points = fe_cache.get_quadrature_points();
  data.local_norm = 0;
  for (unsigned int q=0; q<q_points.size(); ++q)
    {
      const auto &grad_a = grad_a_[q];
      data.local_norm += grad_a*grad_a*JxW[q];
    }
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

  const FEValuesExtractors::Scalar auxiliary_function(1);
  const FEValuesExtractors::Scalar distance(0);
  // const FEValuesExtractors::Scalar abs_gradiente;
  auto &a_t_ = fe_cache.get_values("solution_dot", "a", auxiliary_function, alpha);
  auto &grad_a_ = fe_cache.get_gradients("solution", "grad_a", auxiliary_function, alpha);
  auto &grad_a_e_ = fe_cache.get_gradients("explicit_solution", "grad_a", auxiliary_function, dummy);
  auto &a_ = fe_cache.get_values("solution", "a", auxiliary_function, alpha);

  auto &d_t_ = fe_cache.get_values("solution_dot", "d", distance, alpha);
  auto &grad_d_ = fe_cache.get_gradients("solution", "grad_d", distance, alpha);
  auto &grad_d_e_ = fe_cache.get_gradients("explicit_solution", "grad_d", distance, dummy);
  auto &d_ = fe_cache.get_values("solution", "d", distance, alpha);

  auto &JxW = fe_cache.get_JxW_values();
  auto t = this->get_current_time();
  const unsigned int n_q_points = d_.size();
  // auto &pcout = this->get_pcout();
  auto &pcout = this->get_pcout();
  auto &fev = fe_cache.get_current_fe_values();
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const auto &a = a_[q];
      const auto &a_t = a_t_[q];
      const auto &grad_a = grad_a_[q];
      const auto &grad_a_e = grad_a_e_[q];

      const auto &d = d_[q];
      const auto &grad_d = grad_d_[q];

      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
          auto a_test = fev[auxiliary_function].value(i,q);
          auto grad_a_test = fev[auxiliary_function].gradient(i,q);

          auto d_test = fev[distance].value(i,q);
          auto grad_d_test = fev[distance].gradient(i,q);


          auto direction = grad_a_e;
          if (grad_a_norm > 0 )
            direction = grad_a_e/grad_a_norm;

          auto res = (grad_d - direction ) * grad_d_test;
          // ResidualType res = 0;

          if (use_heat_equation)
            res += a_t * a_test + grad_a * grad_a_test;
          else
            {
              res += (grad_a*grad_a - 1) * a_test;
              res += auxiliary_function_stabilization * grad_a * grad_a_test;
            }

          local_residuals[0][i] += res * JxW[q];
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

  AMG_A.initialize_preconditioner<dim, spacedim>( matrices[0]->block(0,0), fe, dh);
  AMG_B.initialize_preconditioner<dim, spacedim>( matrices[0]->block(1,1), fe, dh);

  LinearOperator<BVEC> A  =
    linear_operator<BVEC>( matrices[0]->block(0,0) );

  LinearOperator<BVEC> B  =
    linear_operator<BVEC>( matrices[0]->block(1,1) );

  LinearOperator<BVEC> C  =
    linear_operator<BVEC>( matrices[0]->block(1,0) );

  LinearOperator<BVEC> D  =
    linear_operator<BVEC>( matrices[0]->block(0,1) );

  LinearOperator<BVEC> A_inv =
    linear_operator<BVEC>( matrices[0]->block(0,0), AMG_A);

  LinearOperator<BVEC> B_inv =
    linear_operator<BVEC>( matrices[0]->block(1,1), AMG_B);

  // ASSEMBLE THE PROBLEM:
  system_op = block_operator<2, 2, VEC>({{
      {{ A, D }} ,
      {{ C, B }}
    }
  });

  typedef LinearOperator<TrilinosWrappers::MPI::Vector,TrilinosWrappers::MPI::Vector> Op_MPI;


  // //////////////////////////////
  BlockLinearOperator<VEC> diag_inv
  = block_diagonal_operator<2, VEC>(std::array<Op_MPI,2>({{ A_inv, B_inv }}));
  prec_op = diag_inv;

  // // Finer preconditioner
  // //////////////////////////////
  // BlockLinearOperator<VEC> M = block_operator<2, 2, VEC>({{
  //     {{ null_operator(A), D               }},
  //     {{ C, null_operator(B) }}
  //   }
  // });
  // BlockLinearOperator<VEC> diag_inv
  // = block_diagonal_operator<2, VEC>({{ A_inv, B_inv }});
  // prec_op = block_back_substitution(M, diag_inv);
}

#endif

/*! @} */
