#ifndef _heat_equation_derived_interface_h_
#define _heat_equation_derived_interface_h_

#include "conservative_interface.h"
#include "parsed_function.h"


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
#include "assembly.h"


typedef TrilinosWrappers::MPI::BlockVector VEC;

template <int dim>
class HeatEquation : public ConservativeInterface<dim,dim,1, HeatEquation<dim> >
{
  typedef Assembly::Scratch::NFields<dim,dim> Scratch;
  typedef Assembly::CopyData::NFieldsPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::NFieldsSystem<dim,dim> CopySystem;
public:


  /* specific and useful functions for this very problem */
  HeatEquation() :
    ConservativeInterface<dim,dim,1,HeatEquation<dim> >("Heat Equation",
                                                        "FESystem[FE_Q(2)]",
                                                        "u", "1", "0")
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
                     Scratch &scratch,
                     CopySystem &data,
                     Number &energy) const
  {
    auto &d = scratch.anydata;
    auto &sol = d.template get<const VEC> ("solution");
    auto &sol_dot = d.template get<const VEC> ("solution_dot");
    Number alpha = d.template get<const double> ("alpha");
    auto &t = d.template get<const double> ("t");

    auto &fe_cache = scratch.fe_cache;

    fe_cache.reinit(cell);

    fe_cache.set_solution_vector("solution", sol, alpha);
    fe_cache.set_solution_vector("solution_dot", sol_dot, alpha);
    this->fix_solution_dot_derivative(fe_cache, alpha);

    auto &JxW = fe_cache.get_JxW_values();

    const FEValuesExtractors::Scalar scalar(0);

    auto &us = fe_cache.get_values("solution", "u", scalar, alpha);
    auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", scalar, alpha);
    auto &grad_us = fe_cache.get_grad_values("solution", "gradu", scalar, alpha);

    energy = 0;
    for (unsigned int q=0; q<us.size(); ++q)
      {
        const Number &u = us[q];
        const Number &u_dot = us_dot[q];
        const Tensor <1, dim, Number> &grad_u = grad_us[q];

        energy += (u_dot*u  + 0.5*(grad_u*grad_u))*JxW[q];
      }
  };

  void compute_system_operators(const DoFHandler<dim> &dh,
                                const TrilinosWrappers::BlockSparseMatrix &matrix,
                                const TrilinosWrappers::BlockSparseMatrix &preconditioner_matrix,
                                LinearOperator<TrilinosWrappers::MPI::BlockVector> &system_op,
                                LinearOperator<TrilinosWrappers::MPI::BlockVector> &prec_op) const
  {

    // std::vector<std::vector<bool> > constant_modes;
    // FEValuesExtractors::Vector velocity_components(0);
    // DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(velocity_components),
    //              constant_modes);

    //    Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
    Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

    TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
    // Amg_data.constant_modes = constant_modes;
    Amg_data.elliptic = true;
    Amg_data.higher_order_elements = true;
    Amg_data.smoother_sweeps = 2;
    Amg_data.aggregation_threshold = 0.02;

    // Mp_preconditioner->initialize (preconditioner_matrix.block(1,1));
    Amg_preconditioner->initialize (matrix.block(0,0),
                                    Amg_data);

    auto A = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,0) );
    system_op  = block_operator<1, 1, TrilinosWrappers::MPI::BlockVector >({{ {{A}} }});
    auto P = linear_operator< TrilinosWrappers::MPI::Vector >(matrix.block(0,0) , *Amg_preconditioner );
    prec_op  =  block_operator<1, 1, TrilinosWrappers::MPI::BlockVector >({{
        {{P}},
      }
    });
  };

private:

  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
};


#endif
