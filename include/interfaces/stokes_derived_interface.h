#ifndef _stokes_derived_interface_h_
#define _stokes_derived_interface_h_

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
class StokesDerivedInterface : public ConservativeInterface<dim,dim,dim+1, StokesDerivedInterface<dim> >
{
  typedef Assembly::Scratch::NFields<dim,dim> Scratch;
  typedef Assembly::CopyData::NFieldsPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::NFieldsSystem<dim,dim> CopySystem;
public:

  /* specific and useful functions for this very problem */
  StokesDerivedInterface();

  virtual void declare_parameters (ParameterHandler &prm);

  template <typename Number>
  void initialize_preconditioner_data(SAKData &d) const;


  template <typename Number>
  void initialize_system_data(SAKData &d) const;


  /* these functions MUST have the follwowing names
   *  because they are called by the ConservativeInterface class
   */
  template <typename Number>
  void prepare_preconditioner_data(const typename DoFHandler<dim,dim>::active_cell_iterator &cell,
                                   Scratch &scratch,
                                   CopyPreconditioner    &data) const;

  template <typename Number>
  void prepare_system_data(const typename DoFHandler<dim,dim>::active_cell_iterator &cell,
                           Scratch &scratch,
                           CopySystem    &data) const;



  template<typename Number>
  void preconditioner_energy(const typename DoFHandler<dim>::active_cell_iterator &,
                             Scratch &,
                             CopyPreconditioner &,
                             Number &energy) const;

  template<typename Number>
  void system_energy(const typename DoFHandler<dim>::active_cell_iterator &,
                     Scratch &,
                     CopySystem &,
                     Number &energy) const;

  virtual void compute_system_operators(const DoFHandler<dim> &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        LinearOperator<TrilinosWrappers::MPI::BlockVector> &,
                                        LinearOperator<TrilinosWrappers::MPI::BlockVector> &) const;

private:
  double eta;
  double rho;
  double nu;

  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

};

template<int dim>
StokesDerivedInterface<dim>::StokesDerivedInterface() :
  ConservativeInterface<dim,dim,dim+1,StokesDerivedInterface<dim> >("Stokes Interface",
      "FESystem[FE_Q(2)^d-FE_Q(1)]",
      "u,u,p", "1,1; 1,0", "1,0; 0,1")
{};


template<int dim>
template<typename Number>
void StokesDerivedInterface<dim>::initialize_preconditioner_data(SAKData &d) const
{
  std::string suffix = typeid(Number).name();
  auto &n_q_points = d.get<unsigned int >("n_q_points");
  auto &dofs_per_cell = d.get<unsigned int >("dofs_per_cell");

  std::vector<Number> independent_local_dof_values (dofs_per_cell);
  std::vector <Number> ps(n_q_points);
  std::vector <Tensor <2, dim, Number> > grad_us(n_q_points);

  d.add_copy(independent_local_dof_values, "independent_local_dof_values"+suffix);

  d.add_copy(grad_us, "grad_us"+suffix);
  d.add_copy(ps, "ps"+suffix);

}

template <int dim>
template<typename Number>
void StokesDerivedInterface<dim>::initialize_system_data(SAKData &d) const
{
  std::string suffix = typeid(Number).name();
  auto &n_q_points = d.get<unsigned int >("n_q_points");
  auto &n_face_q_points = d.get<unsigned int >("n_face_q_points");
  auto &dofs_per_cell = d.get<unsigned int >("dofs_per_cell");

  std::vector<Number> independent_local_dof_values (dofs_per_cell);
  std::vector <Tensor <1, dim, Number> > us(n_q_points);
  std::vector <Tensor <2, dim, Number> > sym_grad_us(n_q_points);
  std::vector <Number> div_us(n_q_points);
  std::vector <Number> ps(n_q_points);
  std::vector <std::vector<Number> > vars(n_q_points,std::vector<Number>(dim+1));
  std::vector <std::vector<Number> > vars_face(n_face_q_points,std::vector<Number>(dim+1));

  d.add_copy(independent_local_dof_values, "independent_local_dof_values"+suffix);
  d.add_copy(us, "us"+suffix);
  d.add_copy(ps, "ps"+suffix);
  d.add_copy(div_us, "div_us"+suffix);
  d.add_copy(sym_grad_us, "sym_grad_us"+suffix);
  d.add_copy(vars, "vars"+suffix);
  d.add_copy(vars_face, "vars_face"+suffix);

}

template <int dim>
template <typename Number>
void StokesDerivedInterface<dim>::prepare_preconditioner_data(const typename DoFHandler<dim,dim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopyPreconditioner    &data) const
{
  std::string suffix = typeid(Number).name();
  auto &sol = scratch.anydata.template get<const TrilinosWrappers::MPI::BlockVector> ("solution");
  auto &sol_dot = scratch.anydata.template get<const TrilinosWrappers::MPI::BlockVector> ("solution_dot");
  auto &t = scratch.anydata.template get<double> ("t");
  auto &alpha = scratch.anydata.template get<double> ("alpha");
  auto &independent_local_dof_values = scratch.anydata.template get<std::vector<Number> >("independent_local_dof_values"+suffix);

  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &grad_us = scratch.anydata.template get<std::vector <Tensor <2, dim, Number> > >("grad_us"+suffix);

  scratch.fe_values.reinit (cell);

  DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values);
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  DOFUtilities::get_grad_values(scratch.fe_values, independent_local_dof_values, velocities, grad_us);
  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, pressure, ps);

}


template <int dim>
template <typename Number>
void StokesDerivedInterface<dim>::prepare_system_data(const typename DoFHandler<dim,dim>::active_cell_iterator &cell,
                                                      Scratch &scratch,
                                                      CopySystem    &data) const
{
  std::string suffix = typeid(Number).name();
  auto &sol = scratch.anydata.template get<const TrilinosWrappers::MPI::BlockVector> ("solution");
  auto &sol_dot = scratch.anydata.template get<const TrilinosWrappers::MPI::BlockVector> ("solution_dot");
  auto &alpha = scratch.anydata.template get<const double> ("alpha");
  auto &t = scratch.anydata.template get<const double> ("t");
  auto &independent_local_dof_values = scratch.anydata.template get<std::vector<Number> >("independent_local_dof_values"+suffix);
  auto &div_us = scratch.anydata.template get<std::vector <Number> >("div_us"+suffix);
  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &us = scratch.anydata.template get<std::vector <Tensor <1, dim, Number> > >("us"+suffix);
  auto &sym_grad_us = scratch.anydata.template get<std::vector <Tensor <2, dim, Number> > >("sym_grad_us"+suffix);

  scratch.fe_values.reinit (cell);

  DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values);
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, velocities, us);
  DOFUtilities::get_div_values(scratch.fe_values, independent_local_dof_values, velocities, div_us);
  DOFUtilities::get_sym_grad_values(scratch.fe_values, independent_local_dof_values, velocities, sym_grad_us);
  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, pressure, ps);

}

template <int dim>
template<typename Number>
void StokesDerivedInterface<dim>::preconditioner_energy(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                        Scratch &scratch,
                                                        CopyPreconditioner &data,
                                                        Number &energy) const
{
  std::string suffix = typeid(Number).name();

  if (scratch.anydata.have("ps"+suffix) == false)
    initialize_preconditioner_data<Number>(scratch.anydata);

  prepare_preconditioner_data<Number>(cell, scratch, data);

  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &grad_us = scratch.anydata.template get<std::vector <Tensor <2, dim, Number> > >("grad_us"+suffix);

  const unsigned int n_q_points = ps.size();

  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Number &p = ps[q];
      const Tensor <2, dim, Number> &grad_u = grad_us[q];

      energy += (eta*.5*scalar_product(grad_u,grad_u) +
                 (1./eta)*0.5*p*p)*scratch.fe_values.JxW(q);
    }
}

template <int dim>
template<typename Number>
void StokesDerivedInterface<dim>::system_energy(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                Scratch &scratch,
                                                CopySystem &data,
                                                Number &energy) const
{
  std::string suffix = typeid(Number).name();

  if (scratch.anydata.have("ps"+suffix) == false)
    initialize_system_data<Number>(scratch.anydata);

  prepare_system_data<Number>(cell, scratch, data);

  auto &us = scratch.anydata.template get<std::vector <Tensor <1, dim, Number> > >("us"+suffix);
  auto &div_us = scratch.anydata.template get<std::vector <Number> > ("div_us"+suffix);
  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &sym_grad_us = scratch.anydata.template get<std::vector <Tensor <2, dim, Number> > >("sym_grad_us"+suffix);

  const unsigned int n_q_points = ps.size();

  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {

      const Tensor <1, dim, Number> &u = us[q];
      const Number &div_u = div_us[q];
      const Number &p = ps[q];
      const Tensor <2, dim, Number> &sym_grad_u = sym_grad_us[q];

      Number psi = (eta*scalar_product(sym_grad_u,sym_grad_u) - p*div_u);
      energy += psi*scratch.fe_values.JxW(q);
    }
}


template <int dim>
void StokesDerivedInterface<dim>::declare_parameters (ParameterHandler &prm)
{
  ParsedFiniteElement<dim,dim>::declare_parameters(prm);
  this->add_parameter(prm, &eta, "eta [Pa s]", "1.0", Patterns::Double(0.0));
}


template <int dim>
void
StokesDerivedInterface<dim>::compute_system_operators(const DoFHandler<dim> &dh,
                                                      const TrilinosWrappers::BlockSparseMatrix &matrix,
                                                      const TrilinosWrappers::BlockSparseMatrix &preconditioner_matrix,
                                                      LinearOperator<TrilinosWrappers::MPI::BlockVector> &system_op,
                                                      LinearOperator<TrilinosWrappers::MPI::BlockVector> &prec_op) const
{

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector velocity_components(0);
  DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(velocity_components),
                                    constant_modes);

  Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.constant_modes = constant_modes;
  Amg_data.elliptic = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 0.02;

  Mp_preconditioner->initialize (preconditioner_matrix.block(1,1));
  Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
                                  Amg_data);


  // SYSTEM MATRIX:
  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,0) );
  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,1) );
  //  auto B =  transpose_operator(Bt);
  auto B     = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,0) );
  auto ZeroP = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,1) );

  auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >( preconditioner_matrix.block(1,1) );

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
  auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);
  auto Schur_inv = inverse_operator( Mp, solver_CG, *Mp_preconditioner);

  auto P00 = A_inv;
  auto P01 = null_operator(Bt.reinit_range_vector);
  auto P10 = Schur_inv * B * A_inv;
  auto P11 = -1 * Schur_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<2, 2, TrilinosWrappers::MPI::BlockVector >({{
      {{ A, Bt }} ,
      {{ B, ZeroP }}
    }
  });


  //const auto S = linear_operator<TrilinosWrappers::MPI::BlockVector>(matrix);

  prec_op = block_operator<2, 2, TrilinosWrappers::MPI::BlockVector >({{
      {{ P00, P01 }} ,
      {{ P10, P11 }}
    }
  });
}


template class StokesDerivedInterface <2>;

#endif
