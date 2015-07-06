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


  virtual UpdateFlags get_preconditioner_flags() const
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

    std::string suffix = typeid(Number).name();
    std::string suffix_double = typeid(double).name();
    auto &n_q_points = d.template get<unsigned int >("n_q_points");
    auto &n_face_q_points = d.template get<unsigned int >("n_face_q_points");
    auto &dofs_per_cell = d.template get<unsigned int >("dofs_per_cell");

    if (!d.have("us"+suffix))
      {
        d.add_copy(std::vector<Number>(dofs_per_cell),"independent_local_dof_values"+suffix);
        d.add_copy(std::vector<Number>(dofs_per_cell),"independent_local_dof_values_dot"+suffix);
        d.add_copy(std::vector<Number>(n_q_points),"us"+suffix);
        d.add_copy(std::vector<Number>(n_q_points),"us_dot"+suffix);
				d.add_copy(std::vector <std::vector<Number> >(n_q_points,std::vector<Number>(1)), "vars"+suffix);
				d.add_copy(std::vector <std::vector<Number> >(n_face_q_points,std::vector<Number>(1)),"vars_face"+suffix);
        d.add_copy(std::vector<Tensor <1, dim, Number> >(n_q_points),"grad_us"+suffix);
        d.add_copy(std::vector<double>(n_q_points),"fs");

        d.add_copy(std::vector<double>(dofs_per_cell),"independent_local_dof_values"+suffix_double);
        d.add_copy(std::vector<double>(dofs_per_cell),"independent_local_dof_values_dot"+suffix_double);
        d.add_copy(std::vector<double>(n_q_points),"us"+suffix_double);
        d.add_copy(std::vector<double>(n_q_points),"us_dot"+suffix_double);
        d.add_copy(std::vector<Tensor <1, dim, double> >(n_q_points),"grad_us"+suffix_double);
      }

    auto &sol = d.template get<const TrilinosWrappers::MPI::BlockVector> ("solution");
    auto &sol_dot = d.template get<const TrilinosWrappers::MPI::BlockVector> ("solution_dot");
    auto &alpha = d.template get<const double> ("alpha");
    auto &t = d.template get<const double> ("t");

    auto &independent_local_dof_values = d.template get<std::vector<Number> >("independent_local_dof_values"+suffix);
    auto &independent_local_dof_values_dot_double = d.template get<std::vector<double> >("independent_local_dof_values_dot"+suffix_double);
    auto &independent_local_dof_values_double = d.template get<std::vector<double> >("independent_local_dof_values"+suffix_double);


    auto &us = d.template get<std::vector <Number> >("us"+suffix);
    auto &us_dot = d.template get<std::vector <Number> >("us_dot"+suffix);

    auto &us_dot_double = d.template get<std::vector <double> >("us_dot"+suffix_double);
    auto &us_double = d.template get<std::vector <double> >("us"+suffix_double);

    auto &grad_us = d.template get<std::vector <Tensor <1, dim, Number> > >("grad_us"+suffix);
    auto &fs =  d.template get<std::vector <double> >("fs");

    scratch.fe_values.reinit (cell);

    DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values);
    if (suffix != suffix_double)
      {
        DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values_double);
      }
    DOFUtilities::extract_local_dofs(sol_dot, data.local_dof_indices, independent_local_dof_values_dot_double);

    const FEValuesExtractors::Scalar scalar(0);

    DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, scalar, us);
    DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values_double, scalar, us_double);
    DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values_dot_double, scalar, us_dot_double);



    if (suffix != suffix_double) // Otherwise us_dot_double and us_dot are the same object
      for (unsigned int i=0; i<us.size(); ++i)
        {
          us_dot[i] = alpha*us[i] + (us_dot_double[i] - alpha*us_double[i]);
        }

    DOFUtilities::get_grad_values(scratch.fe_values, independent_local_dof_values, scalar, grad_us);

    energy = 0;
    for (unsigned int q=0; q<n_q_points; ++q)
      {
        const Number &u = us[q];
        const Number &u_dot = us_dot[q];
        const Tensor <1, dim, Number> &grad_u = grad_us[q];

        energy += (u_dot*u  + 0.5*(grad_u*grad_u))*scratch.fe_values.JxW(q);
      }
  };

  virtual void compute_system_operators(const DoFHandler<dim> &dh,
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
