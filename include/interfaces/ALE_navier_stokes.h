/*! \addtogroup equations
 * @{
 */

/**
 * This interface solves ALE Navier Stokes Equation:
 *
 */

#ifndef _pidomus_ALE_navier_stokes_h_
#define _pidomus_ALE_navier_stokes_h_

#include "pde_system_interface.h"

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>


////////////////////////////////////////////////////////////////////////////////
/// Used functions:

double get_double(Sdouble num)
{
  return num.val();
}

double get_double(double num)
{
  return num;
}

////////////////////////////////////////////////////////////////////////////////
/// Structs and classes:

template <int dim>
struct CopyForce
{
  CopyForce ()
  {};

  ~CopyForce ()
  {};

  CopyForce (const CopyForce &data)
    :
    local_force(data.local_force)
  {};

  Tensor<1, dim, double> local_force;
};

////////////////////////////////////////////////////////////////////////////////
/// ALE Navier Stokes interface:

template <int dim, int spacedim=dim, typename LAC=LATrilinos>
class ALENavierStokes
  :
  public PDESystemInterface<dim,spacedim,ALENavierStokes<dim,spacedim,LAC>, LAC>
{

public:
  ~ALENavierStokes () {};
  ALENavierStokes ();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  template <typename EnergyType, typename ResidualType>
  void
  energies_and_residuals(
    const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    FEValuesCache<dim,spacedim> &scratch,
    std::vector<EnergyType> &energies,
    std::vector<std::vector<ResidualType>> &residuals,
    bool compute_only_system_terms) const;

  void
  assemble_local_forces(
    const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    FEValuesCache<dim,spacedim> &fe_cache,
    CopyForce<dim> &data
  )const;

  void
  compute_system_operators(
    const std::vector<shared_ptr<LATrilinos::BlockMatrix>>,
    LinearOperator<LATrilinos::VectorType> &,
    LinearOperator<LATrilinos::VectorType> &,
    LinearOperator<LATrilinos::VectorType> &) const;

  void
  solution_preprocessing(FEValuesCache<dim,spacedim> &fe_cache) const;

  void
  output_solution (const unsigned int &current_cycle,
                   const unsigned int &step_number) const;

  void
  set_matrix_couplings(std::vector<std::string> &couplings) const;

  virtual UpdateFlags get_face_update_flags() const
  {
    return (update_values |
            update_gradients | /* this is the new entry */
            update_quadrature_points |
            update_normal_vectors |
            update_JxW_values);
  }

private:
  mutable ParsedMappedFunctions<spacedim> nietsche;

  mutable Tensor<1, dim, double> output_force;

// Physical parameter
  double nu;
  double rho;

  double c;
  double k;


  bool use_mass_matrix_for_d_dot;

  bool Mp_use_inverse_operator;

// AMG
  double Amg_data_d_smoother_sweeps;
  double Amg_data_d_aggregation_threshold;
  bool   Amg_d_use_inverse_operator;

  double Amg_data_v_smoother_sweeps;
  double Amg_data_v_aggregation_threshold;
  bool   Amg_v_use_inverse_operator;

  double Amg_data_u_smoother_sweeps;
  double Amg_data_u_aggregation_threshold;
  bool   Amg_u_use_inverse_operator;

// mutable shared_ptr<TrilinosWrappers::PreconditionAMG> amg_A;
  mutable shared_ptr<TrilinosWrappers::PreconditionAMG> P00_preconditioner, P11_preconditioner, P22_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> P33_preconditioner;

  ConditionalOStream pcout;
};


template <int dim, int spacedim, typename LAC>
ALENavierStokes<dim,spacedim, LAC>::
ALENavierStokes()
  :
  PDESystemInterface<dim,spacedim,ALENavierStokes<dim,spacedim,LAC>, LAC>(
    "ALE Navier Stokes Interface",
    dim+dim+dim+1,
    2,
    "FESystem[FE_Q(1)^d-FE_Q(1)^d-FE_Q(2)^d-FE_Q(1)]",
    "d,d,v,v,u,u,p",
    "1,1,1,0"),
  nietsche("Nietsche boundary conditions",
           this->n_components,
           this->get_component_names(),
           "" /* do nothing by default */
          ),
  pcout(std::cout,
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
  this->init();
}

template <int dim, int spacedim, typename LAC>
void ALENavierStokes<dim,spacedim,LAC>::
declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, ALENavierStokes<dim,spacedim,LAC>,LAC>::
  declare_parameters(prm);

  this->add_parameter(prm, &nu,
                      "nu [Pa s]", "1.0",
                      Patterns::Double(0.0),
                      "Viscosity");

  this->add_parameter(prm, &rho,
                      "rho [Kg m^-d]", "1.0",
                      Patterns::Double(0.0),
                      "Density");

  this->add_parameter(prm, &c,
                      "c", "0.0",
                      Patterns::Double(0.0),
                      "c");

  this->add_parameter(prm, &k,
                      "k", "1.0",
                      Patterns::Double(0.0),
                      "k");

  this->add_parameter(prm, &use_mass_matrix_for_d_dot,
                      "Use mass matrix for d_dot", "true",
                      Patterns::Bool(),
                      "Use d_dot mass matrix to close the system \n"
                      "If false it uses the stifness matrix");

  this->add_parameter(prm, &Mp_use_inverse_operator,
                      "Invert Mp using inverse operator", "false",
                      Patterns::Bool(),
                      "Invert Mp usign inverse operator");

  this->add_parameter(prm, &Amg_data_d_smoother_sweeps,
                      "AMG d - smoother sweeps", "2",
                      Patterns::Integer(0),
                      "AMG d - smoother sweeps");

  this->add_parameter(prm, &Amg_data_d_aggregation_threshold,
                      "AMG d - aggregation threshold", "0.02",
                      Patterns::Double(0.0),
                      "AMG d - aggregation threshold");

  this->add_parameter(prm, &Amg_d_use_inverse_operator,
                      "AMG d - use inverse operator", "false",
                      Patterns::Bool(),
                      "Enable the use of inverse operator for AMG d");

  this->add_parameter(prm, &Amg_data_v_smoother_sweeps,
                      "AMG v - smoother sweeps", "2",
                      Patterns::Integer(0),
                      "AMG v - smoother sweeps");

  this->add_parameter(prm, &Amg_data_v_aggregation_threshold,
                      "AMG v - aggregation threshold", "0.02",
                      Patterns::Double(0.0),
                      "AMG v - aggregation threshold");

  this->add_parameter(prm, &Amg_v_use_inverse_operator,
                      "AMG v - use inverse operator", "false",
                      Patterns::Bool(),
                      "Enable the use of inverse operator for AMG v");

  this->add_parameter(prm, &Amg_data_u_smoother_sweeps,
                      "AMG u - smoother sweeps", "2",
                      Patterns::Integer(0),
                      "AMG u - smoother sweeps");

  this->add_parameter(prm, &Amg_data_u_aggregation_threshold,
                      "AMG u - aggregation threshold", "0.02",
                      Patterns::Double(0.0),
                      "AMG u - aggregation threshold");

  this->add_parameter(prm, &Amg_u_use_inverse_operator,
                      "AMG u - use inverse operator", "false",
                      Patterns::Bool(),
                      "Enable the use of inverse operator for AMG u");
}

template <int dim, int spacedim, typename LAC>
void ALENavierStokes<dim,spacedim,LAC>::
parse_parameters_call_back ()
{}

template <int dim, int spacedim, typename LAC>
void ALENavierStokes<dim,spacedim,LAC>::
set_matrix_couplings(std::vector<std::string> &couplings) const
{
  couplings[0] = "1,0,1,1; 0,1,0,0; 1,0,1,1; 1,0,1,0"; // TODO: Select only not null entries
  couplings[1] = "0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,1";
}

template <int dim, int spacedim, typename LAC>
void
ALENavierStokes<dim,spacedim,LAC>::
solution_preprocessing(FEValuesCache<dim,spacedim> &fe_cache) const
{
  Tensor<1, dim, double> global_force;

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;

  auto local_copy = [this, &global_force]
                    (const CopyForce<dim> &data)
  {
    global_force += data.local_force;
  };

  auto local_assemble = [this]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         CopyForce<dim> &data)
  {
    assemble_local_forces(cell, scratch, data);
  };


  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   this->get_dof_handler().begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   this->get_dof_handler().end()),
       local_assemble,
       local_copy,
       fe_cache,
       CopyForce<dim>());

  auto &cache = fe_cache.get_cache();

  global_force[1] = Utilities::MPI::sum(global_force[1],MPI_COMM_WORLD);

  cache.template add_copy<Tensor<1, dim, double> >(global_force, "global_force");

  output_force = global_force;
}

template <int dim, int spacedim, typename LAC>
void
ALENavierStokes<dim,spacedim, LAC>::
assemble_local_forces(
  const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
  FEValuesCache<dim,spacedim> &fe_cache,
  CopyForce<dim> &data
)const
{
  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Vector displacement_velocity(dim);
  const FEValuesExtractors::Vector velocity(2*dim);
  const FEValuesExtractors::Scalar pressure(3*dim);

  double dummy = 0.0;

  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      unsigned int face_id = cell->face(face)->boundary_id();
      if (cell->face(face)->at_boundary() && nietsche.acts_on_id(face_id))
        {
          this->reinit(dummy, cell, face, fe_cache);
// Velocity:
          auto &sym_grad_u_ = fe_cache.get_symmetric_gradients( "explicit_solution", "grad_u", velocity, dummy);
          auto &p_ = fe_cache.get_values( "explicit_solution", "p", pressure, dummy);

          auto &fev = fe_cache.get_current_fe_values();
          auto &q_points = fe_cache.get_quadrature_points();
          auto &JxW = fe_cache.get_JxW_values();

          for (unsigned int q=0; q<q_points.size(); ++q)
            {
              const Tensor<1, dim, double> n = fev.normal_vector(q);

              // velocity:
              const Tensor <2, dim, double> &sym_grad_u = sym_grad_u_[q];
              const double &p = p_[q];

              Tensor <2, dim, double> Id;
              for (unsigned int i = 0; i<dim; ++i)
                Id[i][i] = 1;

              const Tensor <2, dim, double> sigma =
                - p * Id + nu * sym_grad_u;

              Tensor<1, dim, double> force = sigma * n * JxW[q];
              data.local_force[1] -= force[1]; // Minus is due to normal issue..

            } // end loop over quadrature points
          break;
        } // endif face->at_boundary
    } // end loop over faces
}

template<int dim, int spacedim, typename LAC>
void
ALENavierStokes<dim,spacedim,LAC>::
output_solution (const unsigned int &current_cycle,
                 const unsigned int &step_number) const
{
  std::stringstream suffix;
  suffix << "." << current_cycle << "." << step_number;
  this->data_out.prepare_data_output( this->get_dof_handler(),
                                      suffix.str());
  this->data_out.add_data_vector (this->get_locally_relevant_solution(),
                                  this->get_component_names());
  std::vector<std::string> sol_dot_names =
    Utilities::split_string_list(this->get_component_names());
  for (auto &name : sol_dot_names)
    {
      name += "_dot";
    }
  this->data_out.add_data_vector (this->get_locally_relevant_solution_dot(),
                                  print(sol_dot_names, ","));

  this->data_out.write_data_and_clear(this->get_mapping());


  pcout << " Mean force value on the sphere (vertical value): "
        << output_force[1]
        << std::endl;
}

template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
ALENavierStokes<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &residual,
                       bool compute_only_system_terms) const
{
  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Vector displacement_velocity(dim);
  const FEValuesExtractors::Vector velocity(2*dim);
  const FEValuesExtractors::Scalar pressure(3*dim);

  ResidualType et = 0;
  double dummy = 0.0;

  auto &cache = fe_cache.get_cache();
  auto &force  = cache.template get<Tensor<1, dim, double> >("global_force");

  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      unsigned int face_id = cell->face(face)->boundary_id();
      if (cell->face(face)->at_boundary() && nietsche.acts_on_id(face_id))
        {
          this->reinit(et, cell, face, fe_cache);

// Displacement:
          auto &d_ = fe_cache.get_values("solution", "d", displacement, et);
          auto &d_dot_ = fe_cache.get_values( "solution_dot", "d_dot", displacement, et);

// Displacement velocity:
          auto &v_ = fe_cache.get_values("explicit_solution", "grad_v", displacement_velocity, dummy);
          auto &v_dot_ = fe_cache.get_values( "solution_dot", "v_dot", displacement_velocity, et);

// Velocity:
//          auto &grad_u_ = fe_cache.get_gradients("solution", "u", velocity, et);
          auto &u_ = fe_cache.get_values("solution", "grad_u", velocity, et);

// Pressure:
//          auto &grad_p_ = fe_cache.get_gradients("solution", "p", pressure, et);

          auto &fev = fe_cache.get_current_fe_values();
          auto &q_points = fe_cache.get_quadrature_points();
          auto &JxW = fe_cache.get_JxW_values();

          for (unsigned int q=0; q<q_points.size(); ++q)
            {
// Displacement:
              const Tensor<1, dim, ResidualType> &d = d_[q];

              const Tensor<1, dim, ResidualType> &d_dot = d_dot_[q];

// Displacement velocity:
              const Tensor<1, dim, double> &v = v_[q];
              const Tensor<1, dim, ResidualType> &v_dot = v_dot_[q];

// Velocity:
              const Tensor<1, dim, ResidualType> &u = u_[q];

              for (unsigned int i=0; i<residual[0].size(); ++i)
                {
// Test functions:
                  auto d_test = fev[displacement].value(i,q);
                  auto v_test = fev[displacement_velocity].value(i,q);
                  auto u_test = fev[velocity].value(i,q);

                  Tensor<1, dim, double> f;
                  f[1] = force[1];
                  residual[0][i] += (
                                      (v_dot + c * d_dot + k * d - f) * v_test
                                      + (u - v) * u_test
                                      + (d_dot - v) * d_test
                                    )*JxW[q];
                }
            } // end loop over quadrature points
          break;
        } // endif face->at_boundary
    } // end loop over faces

  this->reinit (et, cell, fe_cache);

// displacement:
//  auto &ds = fe_cache.get_values( "solution", "d", displacement, et);
  auto &grad_ds = fe_cache.get_gradients( "solution", "grad_d", displacement, et);
//  auto &div_ds = fe_cache.get_divergences( "solution", "div_d", displacement, et);
  auto &Fs = fe_cache.get_deformation_gradients( "solution", "Fd", displacement, et);
  auto &ds_dot = fe_cache.get_values( "solution_dot", "d_dot", displacement, et);

// Displacement velocity:
  auto &v_ = fe_cache.get_values( "solution", "v", displacement_velocity, et);

// velocity:
  auto &us = fe_cache.get_values( "solution", "u", velocity, et);
  auto &grad_us = fe_cache.get_gradients( "solution", "grad_u", velocity, et);
  auto &div_us = fe_cache.get_divergences( "solution", "div_u", velocity, et);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients( "solution", "u", velocity, et);
  auto &us_dot = fe_cache.get_values( "solution_dot", "u_dot", velocity, et);

// Previous time step solution:
  auto &u_olds = fe_cache.get_values("explicit_solution","u",velocity,dummy);
  //  auto &ds_dot_old = fe_cache.get_values("explicit_solution","d_dot",displacement,dummy);


// pressure:
  auto &ps = fe_cache.get_values( "solution", "p", pressure, et);

// Jacobian:
  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_quad_points = us.size();

  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int quad=0; quad<n_quad_points; ++quad)
    {
// variables:
// velocity:
      const ResidualType &div_u = div_us[quad];
      const Tensor<1, dim, ResidualType> &u_dot = us_dot[quad];
      const Tensor<2, dim, ResidualType> &grad_u = grad_us[quad];
      const Tensor <2, dim, ResidualType> &sym_grad_u = sym_grad_us[quad];
// displacement
      const Tensor<1, dim, ResidualType> &d_dot = ds_dot[quad];
      const Tensor<2, dim, ResidualType> &grad_d = grad_ds[quad];
// Displacement velocity:
      const Tensor<1, dim, ResidualType> &v = v_[quad];

      const Tensor <2, dim, ResidualType> &F = Fs[quad];
      ResidualType J = determinant(F);
      const Tensor <2, dim, ResidualType> &F_inv = invert(F);
      const Tensor <2, dim, ResidualType> &Ft_inv = transpose(F_inv);

// Previous time step solution:
      const Tensor<1, dim, ResidualType> &u_old = u_olds[quad];

// pressure:
      const ResidualType &p = ps[quad];

// others:
      auto J_ale = J; // jacobian of ALE transformation
// auto div_u_ale = (J_ale * (F_inv * u) );
      Tensor <2, dim, ResidualType> Id;
      for (unsigned int i = 0; i<dim; ++i)
        Id[i][i] = p;

      ResidualType my_rho = rho;
      const Tensor <2, dim, ResidualType> sigma =
        - Id + my_rho * ( nu* sym_grad_u * F_inv + ( Ft_inv * transpose(sym_grad_u) ) ) ;

      for (unsigned int i=0; i<residual[0].size(); ++i)
        {
// test functions:
// Displacement velocity:
          auto v_test = fev[displacement_velocity].value(i,quad);
          auto grad_v_test = fev[displacement_velocity].gradient(i,quad);

// velocity:
          auto u_test = fev[velocity].value(i,quad);
          auto grad_v = fev[velocity].gradient(i,quad);
          auto div_v = fev[velocity].divergence(i,quad);

// displacement:
          auto grad_e = fev[displacement].gradient(i,quad);

// pressure:
          auto m = fev[pressure].value(i,quad);
//          auto q = fev[pressure].value(i,quad);

          residual[1][i] +=
            (
              (1./nu)*p*m
            )*JxW[quad];
          residual[0][i] +=
            (
// time derivative term
              rho*scalar_product( u_dot * J_ale , u_test )
//
              + scalar_product( grad_u * ( F_inv * ( u_old - d_dot ) ) * J_ale , u_test )
//
              + scalar_product( J_ale * sigma * Ft_inv, grad_v )
// divergence free constriant
              - div_u * m
// pressure term
              - p * div_v
// Impose armonicity of d and v=d_dot
              + scalar_product( grad_d , grad_e )
            )*JxW[quad];

          if (use_mass_matrix_for_d_dot)
            {
              residual[0][i] += scalar_product( v , v_test )*JxW[quad];
            }
          else
            {
              residual[0][i] += scalar_product( grad_v , grad_v_test )*JxW[quad];
            }

        }
    }

  (void)compute_only_system_terms;
}


template <int dim, int spacedim, typename LAC>
void
ALENavierStokes<dim,spacedim,LAC>::compute_system_operators(
  const std::vector<shared_ptr<LATrilinos::BlockMatrix>> matrices,
  LinearOperator<LATrilinos::VectorType> &system_op,
  LinearOperator<LATrilinos::VectorType> &prec_op,
  LinearOperator<LATrilinos::VectorType> &) const
{
  typedef LATrilinos::VectorType::BlockType BVEC;
  typedef LATrilinos::VectorType VEC;

// AMG data d
  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_d;
  Amg_data_d.elliptic = true;
  // Amg_data_d.higher_order_elements = true;
  Amg_data_d.smoother_sweeps = Amg_data_d_smoother_sweeps;
  Amg_data_d.aggregation_threshold = Amg_data_d_aggregation_threshold;

// AMG data v
  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_v;
  std::vector<std::vector<bool>> constant_modes_v;
  FEValuesExtractors::Vector d_velocity_components(dim);
  DoFTools::extract_constant_modes (
    this->get_dof_handler(),
    this->get_dof_handler().get_fe().component_mask(d_velocity_components),
    constant_modes_v);
  Amg_data_v.constant_modes = constant_modes_v;
  Amg_data_v.elliptic = true;
  // Amg_data_v.higher_order_elements = true;
  Amg_data_v.smoother_sweeps = Amg_data_v_smoother_sweeps;
  Amg_data_v.aggregation_threshold = Amg_data_v_aggregation_threshold;

// AMG data u
  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_u;
  std::vector<std::vector<bool>> constant_modes_u;
  FEValuesExtractors::Vector velocity_components(2*dim);
  DoFTools::extract_constant_modes (
    this->get_dof_handler(),
    this->get_dof_handler().get_fe().component_mask(velocity_components),
    constant_modes_u);
  Amg_data_u.constant_modes = constant_modes_u;
  // Amg_data_u.elliptic = false;
  Amg_data_u.higher_order_elements = true;
  Amg_data_u.smoother_sweeps = Amg_data_u_smoother_sweeps;
  Amg_data_u.aggregation_threshold = Amg_data_u_aggregation_threshold;

// Preconditioners:

  P00_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
  P00_preconditioner ->initialize (matrices[0]->block(0,0), Amg_data_d);
  P11_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
  P11_preconditioner ->initialize (matrices[0]->block(1,1), Amg_data_v);
  P22_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
  P22_preconditioner ->initialize (matrices[0]->block(2,2), Amg_data_u);
  P33_preconditioner.reset (new TrilinosWrappers::PreconditionJacobi());
  P33_preconditioner ->initialize (matrices[1]->block(3,3), 1.4);

////////////////////////////////////////////////////////////////////////////
// SYSTEM MATRIX:

  std::array<std::array<LinearOperator< BVEC >, 4>, 4> S;
  for (unsigned int i = 0; i<4; ++i)
    for (unsigned int j = 0; j<4; ++j)
      S[i][j] = linear_operator< BVEC >(matrices[0]->block(i,j) );
  system_op = BlockLinearOperator< VEC >(S);

////////////////////////////////////////////////////////////////////////////
// PRECONDITIONER MATRIX:

  std::array<std::array<LinearOperator<TrilinosWrappers::MPI::Vector>, 4>, 4> P;
  for (unsigned int i = 0; i<4; ++i)
    for (unsigned int j = 0; j<4; ++j)
      P[i][j] = linear_operator<BVEC>( matrices[0]->block(i,j) );

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<BVEC> solver_CG(solver_control_pre);
  static SolverGMRES<BVEC> solver_GMRES(solver_control_pre);



  for (unsigned int i = 0; i<4; ++i)
    for (unsigned int j = 0; j<4; ++j)
      if (i!=j)
        P[i][j] = null_operator< TrilinosWrappers::MPI::Vector >(P[i][j]);

  auto A = linear_operator< BVEC>(matrices[0]->block(2,2) );
  auto B = linear_operator< BVEC>(matrices[0]->block(3,2) );
  auto Bt = transpose_operator< >(B);

  LinearOperator<BVEC> A_inv;
  if (Amg_v_use_inverse_operator)
    {
      A_inv = inverse_operator( S[2][2],
                                solver_GMRES,
                                *P22_preconditioner);
    }
  else
    {
      A_inv = linear_operator<BVEC>(matrices[0]->block(2,2),
                                    *P22_preconditioner);
    }

  auto Mp = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[1]->block(3,3) );

  LinearOperator<BVEC> Mp_inv;
  if (Mp_use_inverse_operator)
    {
      Mp_inv = inverse_operator(Mp,
                                solver_GMRES,
                                *P33_preconditioner);
    }
  else
    {
      Mp_inv = linear_operator<BVEC>(matrices[1]->block(3,3),
                                     *P33_preconditioner);
    }

  auto Schur_inv = nu * Mp_inv;

  if (Amg_d_use_inverse_operator)
    {
      P[0][0] = inverse_operator( S[0][0],
                                  solver_CG,
                                  *P00_preconditioner);
    }
  else
    {
      P[0][0] = linear_operator<BVEC>(matrices[0]->block(0,0),
                                      *P00_preconditioner);
    }

  if (Amg_v_use_inverse_operator)
    {
      P[1][1] = inverse_operator( S[1][1],
                                  solver_CG,
                                  *P11_preconditioner);
    }
  else
    {
      P[1][1] = linear_operator<BVEC>(matrices[0]->block(1,1),
                                      *P11_preconditioner);
    }

  P[2][2] = A_inv;
  P[2][3] = A_inv * Bt * Schur_inv;
  P[3][2] = null_operator(B);
  P[3][3] = -1 * Schur_inv;


  prec_op = BlockLinearOperator< VEC >(P);
}

#endif

/*! @} */
