#ifndef _compressible_neo_hookean_h_
#define _compressible_neo_hookean_h_

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
class CompressibleNeoHookeanInterface : public ConservativeInterface<dim,spacedim,dim, CompressibleNeoHookeanInterface<dim,spacedim> >
{
  typedef Assembly::Scratch::NFields<dim,spacedim> Scratch;
  typedef Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> CopyPreconditioner;
  typedef Assembly::CopyData::NFieldsSystem<dim,spacedim> CopySystem;
public:


  /* specific and useful functions for this very problem */
  CompressibleNeoHookeanInterface();

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
CompressibleNeoHookeanInterface<dim,spacedim>::CompressibleNeoHookeanInterface() :
  ConservativeInterface<dim,spacedim,dim,CompressibleNeoHookeanInterface<dim,spacedim> >("Compressible NeoHookean Interface",
      "FESystem[FE_Q(1)^d]",
      "u,u,u", "1", "1")
{};


template <int dim, int spacedim>
template<typename Number>
void CompressibleNeoHookeanInterface<dim,spacedim>::initialize_preconditioner_data(SAKData &d) const
{
  std::string suffix = typeid(Number).name();
  auto &n_q_points = d.get<unsigned int >("n_q_points");
  auto &n_face_q_points = d.get<unsigned int >("n_face_q_points");
  auto &dofs_per_cell = d.get<unsigned int >("dofs_per_cell");

  std::vector<Number> independent_local_dof_values (dofs_per_cell);
  std::vector <Tensor <1, dim, Number> > us(n_q_points);

  d.add_copy(independent_local_dof_values, "independent_local_dof_values"+suffix);
  d.add_copy(us, "us"+suffix);

}

template <int dim, int spacedim>
template<typename Number>
void CompressibleNeoHookeanInterface<dim,spacedim>::initialize_system_data(SAKData &d) const
{
  std::string suffix = typeid(Number).name();
	std::string suffix_double = typeid(double).name();
  auto &n_q_points = d.get<unsigned int >("n_q_points");
  auto &n_face_q_points = d.get<unsigned int >("n_face_q_points");
  auto &dofs_per_cell = d.get<unsigned int >("dofs_per_cell");

  std::vector<Number> independent_local_dof_values (dofs_per_cell);
  std::vector <Tensor <1, dim, Number> > us(n_q_points);
  std::vector <std::vector<Number> > vars(n_q_points,std::vector<Number>(dim));
  std::vector <std::vector<Number> > vars_face(n_face_q_points,std::vector<Number>(dim));
  std::vector <Tensor <1, spacedim, Number> > us_face(n_face_q_points);
  std::vector <Tensor <2, spacedim, Number> > Fs(n_q_points);

  d.add_copy(independent_local_dof_values, "independent_local_dof_values"+suffix);
  d.add_copy(us, "us"+suffix);
  d.add_copy(us_face, "us_face"+suffix);
  d.add_copy(Fs, "Fs"+suffix);
  d.add_copy(vars, "vars"+suffix);
  d.add_copy(vars_face, "vars_face"+suffix);
	d.add_copy(std::vector<Number>(dofs_per_cell),"independent_local_dof_values_dot"+suffix);
	d.add_copy(std::vector <Tensor <1, dim, Number> >(n_q_points),"us_dot"+suffix);
	d.add_copy(std::vector<double>(dofs_per_cell),"independent_local_dof_values"+suffix_double);
	d.add_copy(std::vector<double>(dofs_per_cell),"independent_local_dof_values_dot"+suffix_double);
	d.add_copy(std::vector <Tensor <1, dim, double> >(n_q_points),"us"+suffix_double);
	d.add_copy(std::vector <Tensor <1, dim, double> >(n_q_points),"us_dot"+suffix_double);

}

template <int dim, int spacedim>
template <typename Number>
void CompressibleNeoHookeanInterface<dim,spacedim>::prepare_preconditioner_data(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopyPreconditioner    &data) const
{
  std::string suffix = typeid(Number).name();
  auto &sol = scratch.anydata.template get<const TrilinosWrappers::MPI::BlockVector> ("solution");
  auto &independent_local_dof_values = scratch.anydata.template get<std::vector<Number> >("independent_local_dof_values"+suffix);
  auto &us = scratch.anydata.template get<std::vector <Tensor <1, dim, Number> > >("us"+suffix);

  scratch.fe_values.reinit (cell);

  DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values);
  const FEValuesExtractors::Vector displacement(0);

  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, displacement, us);

}


template <int dim, int spacedim>
template <typename Number>
void CompressibleNeoHookeanInterface<dim,spacedim>::prepare_system_data(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopySystem    &data) const
{
  std::string suffix = typeid(Number).name();
	std::string suffix_double = typeid(double).name();
	auto &d = scratch.anydata;
  auto &sol = d.template get<const TrilinosWrappers::MPI::BlockVector> ("solution");
	auto &sol_dot = d.template get<const TrilinosWrappers::MPI::BlockVector> ("solution_dot");
	auto &alpha = d.template get<const double> ("alpha");
	auto &t = d.template get<const double> ("t");
	auto &independent_local_dof_values = d.template get<std::vector<Number> >("independent_local_dof_values"+suffix);
	auto &independent_local_dof_values_dot_double = d.template get<std::vector<double> >("independent_local_dof_values_dot"+suffix_double);
	auto &independent_local_dof_values_double = d.template get<std::vector<double> >("independent_local_dof_values"+suffix_double);

	auto &us = d.template get<std::vector <Tensor <1, dim, Number> > >("us"+suffix);
	auto &us_dot = d.template get<std::vector <Tensor <1, dim, Number> > >("us_dot"+suffix);

	auto &us_dot_double = d.template get<std::vector <Tensor <1, dim, double> > >("us_dot"+suffix_double);
	auto &us_double = d.template get<std::vector <Tensor <1, dim, double> > >("us"+suffix_double);

  auto &Fs = d.template get<std::vector <Tensor <2, dim, Number> > >("Fs"+suffix);

  scratch.fe_values.reinit (cell);

  DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values);
	if (suffix != suffix_double)
	{
		DOFUtilities::extract_local_dofs(sol, data.local_dof_indices, independent_local_dof_values_double);
	}
	DOFUtilities::extract_local_dofs(sol_dot, data.local_dof_indices, independent_local_dof_values_dot_double);



  const FEValuesExtractors::Vector displacement(0);

  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values, displacement, us);
  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values_double, displacement, us_double);
  DOFUtilities::get_values(scratch.fe_values, independent_local_dof_values_dot_double, displacement, us_dot_double);
  DOFUtilities::get_F_values(scratch.fe_values, independent_local_dof_values, displacement, Fs);
	if (suffix != suffix_double) // Otherwise us_dot_double and us_dot are the same object
		for (unsigned int i=0; i<us.size(); ++i)
			for (unsigned int d=0; d<dim; ++d)
				us_dot[i][d] = alpha*us[i][d] + (us_dot_double[i][d]- alpha*us_double[i][d]);

}

template <int dim, int spacedim>
template<typename Number>
void CompressibleNeoHookeanInterface<dim,spacedim>::preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopyPreconditioner &data,
    Number &energy) const
{
  std::string suffix = typeid(Number).name();

  if (scratch.anydata.have("us"+suffix) == false)
    initialize_preconditioner_data<Number>(scratch.anydata);

  prepare_preconditioner_data<Number>(cell, scratch, data);

  auto &us = scratch.anydata.template get<std::vector <Tensor <1, dim, Number> > >("us"+suffix);

  const unsigned int n_q_points = us.size();

  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor <1, dim, Number> &u = us[q];

      energy += 0.5*(u*u)*scratch.fe_values.JxW(q);
    }
}

template <int dim, int spacedim>
template<typename Number>
void CompressibleNeoHookeanInterface<dim,spacedim>::system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    Scratch &scratch,
    CopySystem &data,
    Number &energy) const
{
  std::string suffix = typeid(Number).name();

  if (scratch.anydata.have("us"+suffix) == false)
    initialize_system_data<Number>(scratch.anydata);

  prepare_system_data<Number>(cell, scratch, data);



  auto &us = scratch.anydata.template get<std::vector <Tensor <1, dim, Number> > >("us"+suffix);
  auto &us_dot = scratch.anydata.template get<std::vector <Tensor <1, dim, Number> > >("us_dot"+suffix);
  auto &Fs = scratch.anydata.template get<std::vector <Tensor <2, dim, Number> > >("Fs"+suffix);

  const unsigned int n_q_points = us.size();

  energy = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {

      const Tensor <1, dim, Number> &u = us[q];
      const Tensor <1, dim, Number> &u_dot = us[q];
      const Tensor <2, dim, Number> &F = Fs[q];
      const Tensor<2, dim, Number> C = transpose(F)*F;

      Number Ic = trace(C);
      Number J = determinant(F);
      Number lnJ = std::log (J);

      Number psi = (mu/2.)*(Ic-dim) - mu*lnJ + (lambda/2.)*(lnJ)*(lnJ);

      energy += (u*u_dot + psi)*scratch.fe_values.JxW(q);

    }

}


template <int dim, int spacedim>
void CompressibleNeoHookeanInterface<dim,spacedim>::declare_parameters (ParameterHandler &prm)
{
  ParsedFiniteElement<dim,spacedim>::declare_parameters(prm);
  this->add_parameter(prm, &E, "Young's modulus", "10.0", Patterns::Double(0.0));
  this->add_parameter(prm, &nu, "Poisson's ratio", "0.3", Patterns::Double(0.0));
}

template <int dim, int spacedim>
void CompressibleNeoHookeanInterface<dim,spacedim>::parse_parameters_call_back ()
{
  mu = E/(2.0*(1.+nu));
  lambda = (E *nu)/((1.+nu)*(1.-2.*nu));
}

template <int dim, int spacedim>
void
CompressibleNeoHookeanInterface<dim,spacedim>::compute_system_operators(const DoFHandler<dim,spacedim> &dh,
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

//  Mp_preconditioner->initialize (preconditioner_matrix.block(0,0));
  Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
                                  Amg_data);


  // SYSTEM MATRIX:
  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,0) );
//  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,1) );
//  //  auto B =  transpose_operator(Bt);
//  auto B     = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,0) );
//  auto ZeroP = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,1) );

//  auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >( preconditioner_matrix.block(0,0) );

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
  auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);
//  auto Schur_inv = inverse_operator( Mp, solver_CG, *Mp_preconditioner);

  auto P00 = A_inv;
//  auto P01 = null_operator(Bt.reinit_range_vector);
//  auto P10 = Schur_inv * B * A_inv;
//  auto P11 = -1 * Schur_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<1, 1, TrilinosWrappers::MPI::BlockVector >({{
      {{ A }}
    }
  });


  //const auto S = linear_operator<TrilinosWrappers::MPI::BlockVector>(matrix);

  prec_op = block_operator<1, 1, TrilinosWrappers::MPI::BlockVector >({{
      {{ P00}} ,
    }
  });
}


template class CompressibleNeoHookeanInterface <3,3>;

#endif
