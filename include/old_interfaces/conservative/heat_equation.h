/*! \addtogroup equations
 *  @{
 */

#ifndef _heat_equation_derived_interface_h_
#define _heat_equation_derived_interface_h_

#include "interfaces/conservative.h"
#include <deal2lkit/parsed_function.h>


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
#include "data/assembly.h"
#include "lac/lac_type.h"


template <int dim, typename LAC=LADealII>
class HeatEquation : public ConservativeInterface<dim,dim,1, HeatEquation<dim,LAC>, LAC >
{
  typedef FEValuesCache<dim,dim> Scratch;
  typedef Assembly::CopyData::piDoMUSPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::piDoMUSSystem<dim,dim> CopySystem;
  typedef BlockSparseMatrix<double> MAT;

public:


  /* specific and useful functions for this very problem */
  HeatEquation() :
    ConservativeInterface<dim,dim,1,HeatEquation<dim,LAC>, LAC >("Heat Equation",
        "FESystem[FE_Q(2)]",
        "u", "1", "0","1")
  {};


  UpdateFlags get_preconditioner_flags() const
  {
    return update_default;
  };


  template<typename Number>
  void preconditioner_energy(const typename DoFHandler<dim>::active_cell_iterator &,
                             Scratch &,
                             CopyPreconditioner &,
                             Number &) const
  {
  };

  template<typename Number>
  void system_energy(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     Scratch &fe_cache,
                     CopySystem &data,
                     Number &energy) const
  {
    Number alpha = this->alpha;
    this->reinit(alpha, cell, fe_cache);

    auto &JxW = fe_cache.get_JxW_values();

    const FEValuesExtractors::Scalar scalar(0);

    auto &us = fe_cache.get_values("solution", "u", scalar, alpha);
    auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", scalar, alpha);
    auto &grad_us = fe_cache.get_gradients("solution", "gradu", scalar, alpha);

    energy = 0;
    for (unsigned int q=0; q<us.size(); ++q)
      {
        const Number &u = us[q];
        const Number &u_dot = us_dot[q];
        const Tensor <1, dim, Number> &grad_u = grad_us[q];

        energy += (u_dot*u  + 0.5*(grad_u*grad_u))*JxW[q];
      }
  };
};


#endif
/*! @} */
