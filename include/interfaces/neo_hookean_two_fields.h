#ifndef _neo_hookean_two_fields_interface_h_
#define _neo_hookean_two_fields_interface_h_

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


template <int dim, int spacedim>
class NeoHookeanTwoFieldsInterface : public ConservativeInterface<dim,spacedim,dim+1, NeoHookeanTwoFieldsInterface<dim,spacedim> >
{
  typedef Assembly::Scratch::NFields<dim,spacedim> Scratch;
  typedef Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> CopyPreconditioner;
  typedef Assembly::CopyData::NFieldsSystem<dim,spacedim> CopySystem;
public:

  /* specific and useful functions for this very problem */
  NeoHookeanTwoFieldsInterface();

  virtual void declare_parameters (ParameterHandler &prm);
  virtual void parse_parameters_call_back ();

  template <typename Number>
  void initialize_preconditioner_data(SAKData &d) const;


  template <typename Number>
  void initialize_system_data(SAKData &d) const;


  /* these functions MUST have the follwowing names
   *  because they are called by the ConservativeInterface class
   */
  template <typename Number>
  void prepare_preconditioner_data(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                   Scratch &scratch,
                                   CopyPreconditioner    &data) const;

  template <typename Number>
  void prepare_system_data(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                           Scratch &scratch,
                           CopySystem    &data) const;



  template<typename Number>
  void preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                             Scratch &,
                             CopyPreconditioner &,
                             Number &energy) const;

  template<typename Number>
  void system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                     Scratch &,
                     CopySystem &,
                     Number &energy) const;

  virtual void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        LinearOperator<TrilinosWrappers::MPI::BlockVector> &,
                                        LinearOperator<TrilinosWrappers::MPI::BlockVector> &) const;

private:
  double E;
  double nu;
  double mu;
  double lambda;

  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

};

template <int dim, int spacedim>
NeoHookeanTwoFieldsInterface<dim,spacedim>::NeoHookeanTwoFieldsInterface() :
  ConservativeInterface<dim,spacedim,dim+1,NeoHookeanTwoFieldsInterface<dim,spacedim> >("NeoHookean Interface",
      "FESystem[FE_Q(2)^d-FE_DGP(1)]",
      "u,u,u,p", "1,1; 1,0", "1,0; 0,1")
{};



template <int dim, int spacedim>
template<typename Number>
void NeoHookeanTwoFieldsInterface<dim,spacedim>::initialize_preconditioner_data(SAKData &d) const
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

template <int dim, int spacedim>
template<typename Number>
void NeoHookeanTwoFieldsInterface<dim,spacedim>::initialize_system_data(SAKData &d) const
{
  std::string suffix = typeid(Number).name();
	std::string suffix_double = typeid(double).name();
  auto &n_q_points = d.get<unsigned int >("n_q_points");
  auto &n_face_q_points = d.get<unsigned int >("n_face_q_points");
  auto &dofs_per_cell = d.get<unsigned int >("dofs_per_cell");

  std::vector<Number> independent_local_dof_values (dofs_per_cell);
  std::vector <std::vector<Number> > vars(n_q_points,std::vector<Number>(dim+1));
  std::vector <std::vector<Number> > vars_face(n_face_q_points,std::vector<Number>(dim+1));
  std::vector <Tensor <1, dim, Number> > us(n_q_points);
  std::vector <Tensor <1, spacedim, Number> > us_face(n_face_q_points);
  std::vector <Tensor <2, spacedim, Number> > Fs(n_q_points);
  std::vector <Number> ps(n_q_points);

  d.add_copy(independent_local_dof_values, "independent_local_dof_values"+suffix);
  d.add_copy(us, "us"+suffix);
  d.add_copy(us_face, "us_face"+suffix);
  d.add_copy(ps, "ps"+suffix);
  d.add_copy(Fs, "Fs"+suffix);
  d.add_copy(vars, "vars"+suffix);
  d.add_copy(vars_face, "vars_face"+suffix);
	d.add_copy(std::vector <Tensor <1, dim, Number> >(n_q_points),"us_dot"+suffix);
	d.add_copy(std::vector<double>(dofs_per_cell),"independent_local_dof_values"+suffix_double);
	d.add_copy(std::vector<double>(dofs_per_cell),"independent_local_dof_values_dot"+suffix_double);
	d.add_copy(std::vector <Tensor <1, dim, double> >(n_q_points),"us"+suffix_double);
	d.add_copy(std::vector <Tensor <1, dim, double> >(n_q_points),"us_dot"+suffix_double);

}

template <int dim, int spacedim>
template <typename Number>
void NeoHookeanTwoFieldsInterface<dim,spacedim>::prepare_preconditioner_data(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopyPreconditioner    &data) const
{
  std::string suffix = typeid(Number).name();
  auto &sol = scratch.anydata.template get<const TrilinosWrappers::MPI::BlockVector> ("solution");
  auto &independent_local_dof_values = scratch.anydata.template get<std::vector<Number> >("independent_local_dof_values"+suffix);

  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &grad_us = scratch.anydata.template get<std::vector <Tensor <2, dim, Number> > >("grad_us"+suffix);

  scratch.fe_values.reinit (cell);

  DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values);
  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Scalar pressure(dim);

  DOFUtilities::get_grad_values(scratch.fe_values, independent_local_dof_values, displacement, grad_us);
  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, pressure, ps);

}


template <int dim, int spacedim>
template <typename Number>
void NeoHookeanTwoFieldsInterface<dim,spacedim>::prepare_system_data(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopySystem    &data) const
{
  std::string suffix = typeid(Number).name();
	std::string suffix_double = typeid(double).name();
  auto &sol = scratch.anydata.template get<const TrilinosWrappers::MPI::BlockVector> ("solution");
  auto &independent_local_dof_values = scratch.anydata.template get<std::vector<Number> >("independent_local_dof_values"+suffix);
  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &us = scratch.anydata.template get<std::vector <Tensor <1, dim, Number> > >("us"+suffix);
  auto &Fs = scratch.anydata.template get<std::vector <Tensor <2, dim, Number> > >("Fs"+suffix);

  scratch.fe_values.reinit (cell);

  DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values);
  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Scalar pressure(dim);

  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, displacement, us);
  DOFUtilities::get_F_values(scratch.fe_values, independent_local_dof_values, displacement, Fs);
  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, pressure, ps);

}

template <int dim, int spacedim>
template<typename Number>
void NeoHookeanTwoFieldsInterface<dim,spacedim>::preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
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

      energy += (scalar_product(grad_u,grad_u) +
                 0.5*p*p)*scratch.fe_values.JxW(q);
    }
}

template <int dim, int spacedim>
template<typename Number>
void NeoHookeanTwoFieldsInterface<dim,spacedim>::system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopySystem &data,
    Number &energy) const
{
  std::string suffix = typeid(Number).name();

  if (scratch.anydata.have("ps"+suffix) == false)
    initialize_system_data<Number>(scratch.anydata);

  prepare_system_data<Number>(cell, scratch, data);



  auto &us = scratch.anydata.template get<std::vector <Tensor <1, dim, Number> > >("us"+suffix);
  auto &ps = scratch.anydata.template get<std::vector <Number> >("ps"+suffix);
  auto &Fs = scratch.anydata.template get<std::vector <Tensor <2, dim, Number> > >("Fs"+suffix);

  const unsigned int n_q_points = ps.size();

  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      Tensor <1, dim, double> B;

      const Tensor <1, dim, Number> &u = us[q];
      const Number &p = ps[q];
      const Tensor <2, dim, Number> &F = Fs[q];
      const Tensor<2, dim, Number> C = transpose(F)*F;

      Number Ic = trace(C);
      Number J = determinant(F);

      Number psi = (mu/2.)*(Ic-dim) +p*(J-1.);
      energy += psi*scratch.fe_values.JxW(q);

    }
}


template <int dim, int spacedim>
void NeoHookeanTwoFieldsInterface<dim,spacedim>::declare_parameters (ParameterHandler &prm)
{
  ParsedFiniteElement<dim,spacedim>::declare_parameters(prm);
  this->add_parameter(prm, &E, "Young's modulus", "10.0", Patterns::Double(0.0));
  this->add_parameter(prm, &nu, "Poisson's ratio", "0.3", Patterns::Double(0.0));
}

template <int dim, int spacedim>
void NeoHookeanTwoFieldsInterface<dim,spacedim>::parse_parameters_call_back ()
{
  mu = E/(2.0*(1.+nu));
  lambda = (E *nu)/((1.+nu)*(1.-2.*nu));
}

template <int dim, int spacedim>
void
NeoHookeanTwoFieldsInterface<dim,spacedim>::compute_system_operators(const DoFHandler<dim,spacedim> &dh,
    const TrilinosWrappers::BlockSparseMatrix &matrix,
    const TrilinosWrappers::BlockSparseMatrix &preconditioner_matrix,
    LinearOperator<TrilinosWrappers::MPI::BlockVector> &system_op,
    LinearOperator<TrilinosWrappers::MPI::BlockVector> &prec_op) const
{

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector displacement(0);
  DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(displacement),
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
  auto P01 = null_operator(Bt);
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


template class NeoHookeanTwoFieldsInterface <3,3>;

#endif
