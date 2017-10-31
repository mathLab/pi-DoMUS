/*! \addtogroup equations
 *  @{
 */

/**
 *  This interface solves a Navier Stokes Equation:
 *  \f[
 *     \begin{cases}
 *       \partial_t u + (u\cdot\nabla)u - \nu\textrm{div} \epsilon(u)
 *     + \frac{1}{\rho}\nabla p = f \\
 *       \textrm{div}u=0
 *     \end{cases}
 *  \f]
 *  where \f$ \epsilon(u) = \frac{\nabla u + [\nabla u]^t}{2}. \f$
 *
 *  Non time-depending Navier Stokes Equation:
 *  \f[
 *     \begin{cases}
 *       (u\cdot\nabla)u - \nu\textrm{div} \epsilon(u)
 *     + \frac{1}{\rho}\nabla p = f \\
 *       \textrm{div}u=0
 *     \end{cases}
 *  \f]
 *  can be recoverd setting @p dynamic = false
 *
 *
 * In the code we adopt the following notations:
 * - Mp := block resulting from \f$ ( \partial_t p, q ) \f$
 * - Ap := block resulting from \f$ \nu ( \nabla p,\nabla q ) \f$
 * - Np := block resulting from \f$ ( u \cdot \nabla p, q) \f$
 *
 * where:
 * - p = pressure
 * - q = test function for the pressure
 * - u = velocity
 * - v = test function for the velocity
 *
 * Notes on preconditioners:
 * - default: This preconditioner uses the mass matrix of pressure block as
 * inverse for the Schur block.
 * This is a preconditioner suitable for problems wher the viscosity is
 * higher than the density. \f[ S^-1 = \nu M_p \f]
 * - identity: Identity matrix preconditioner
 * - low-nu: This preconditioner uses the stifness matrix of pressure block
 * as inverse for the Schur block. \f[ S^-1 = \rho \frac{1}{\Delta t} A_p \f]
 * - cah-cha:  Preconditioner suggested by J. Cahouet and J.-P. Chabard.
 *  \f[ S^-1 =  \nu M_p  + \rho \frac{1}{\Delta t} A_p. \f]
 *
 */

#ifndef _pidomus_navier_stokes_h_
#define _pidomus_navier_stokes_h_

#include "pde_system_interface.h"

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal2lkit/sacado_tools.h>
#include <deal2lkit/parsed_preconditioner/amg.h>
#include <deal2lkit/parsed_preconditioner/jacobi.h>

namespace NSUtilities
{

  ////////////////////////////////////////////////////////////////////////////////
  /// Structs and classes:

  template <int dim>
  struct CopyForce
  {
    CopyForce ()
    {}

    ~CopyForce ()
    {}

    CopyForce (const CopyForce &data)
      :
      local_force(data.local_force)
    {}

    Tensor<1, dim, double> local_force;
  };

}

////////////////////////////////////////////////////////////////////////////////
/// Navier Stokes interface:

template <int dim, int spacedim=dim, typename LAC=LATrilinos>
class NavierStokes
  :
  public PDESystemInterface<dim,spacedim,NavierStokes<dim,spacedim,LAC>, LAC>
{

public:
  virtual ~NavierStokes () {}
  NavierStokes (bool dynamic);

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
  compute_system_operators(
    const std::vector<shared_ptr<LATrilinos::BlockMatrix>> &,
    LinearOperator<LATrilinos::VectorType> &,
    LinearOperator<LATrilinos::VectorType> &,
    LinearOperator<LATrilinos::VectorType> &) const;

  void
  set_matrix_couplings(std::vector<std::string> &couplings) const;

  void
  solution_preprocessing(FEValuesCache<dim,spacedim> &fe_cache) const;

  void
  output_solution (const unsigned int &current_cycle,
                   const unsigned int &step_number) const;

  virtual UpdateFlags get_face_update_flags() const
  {
    return (update_values |
            update_gradients | /* this is the new entry */
            update_quadrature_points |
            update_normal_vectors |
            update_JxW_values);
  }

private:

  /**
   * AMG preconditioner for velocity-velocity matrix.
   */
  mutable ParsedAMGPreconditioner AMG_A;

  /**
   * AMG preconditioner for the pressure stifness matrix.
   */
  mutable ParsedAMGPreconditioner AMG_Ap;

  /**
   * Jacobi preconditioner for the pressure mass matrix.
   */
  mutable ParsedJacobiPreconditioner jac_Mp;

  /**
   * Force acting on the obstables.
   **/
  mutable Tensor<1, dim, double> output_force;

  /**
   * Enable dynamic term: \f$ \partial_t u\f$.
   */
  bool dynamic;

  /**
   * TODO:
   */
  bool use_skew_symmetric_advection;

  /**
   * Hot to handle \f$ (\nabla u)u \f$.
   */
  std::string non_linear_term;

  /**
   * Hot to handle \f$ (\nabla u)u \f$.
   */
  bool linearize_in_time;

  /**
   * Compute the force on the obstable.
   */
  bool compute_force;

  /**
  * Name of the preconditioner:
  */
  std::string prec_name;

  // PHYSICAL PARAMETERS:
  ////////////////////////////////////////////

  /**
   * Density
   */
  double rho;

  /**
   * Viscosity
   */
  double nu;

  // STABILIZATION:
  ////////////////////////////////////////////
  /**
   * div-grad stabilization parameter
   */
  double gamma;

  /**
   * SUPG stabilization term
   */
  double SUPG_alpha;

  /**
   * p-q stabilization parameter
   */
  double gamma_p;

  /**
   * Compute Mp
   */
  bool compute_Mp;

  /**
   * Compute Ap
   */
  bool compute_Ap;

  /**
   * Invert Mp using inverse_operator
   */
  bool invert_Mp;

  /**
   * Invert Ap using inverse_operator
   */
  bool invert_Ap;

  /**
  * Solver tolerance for CG
  */
  double CG_solver_tolerance;

  /**
   * Solver tolerance for GMRES
   */
  double GMRES_solver_tolerance;

  bool is_parallel;
};

template <int dim, int spacedim, typename LAC>
NavierStokes<dim,spacedim, LAC>::
NavierStokes(bool dynamic)
  :
  PDESystemInterface<dim,spacedim,NavierStokes<dim,spacedim,LAC>, LAC>(
    "Navier Stokes Interface",
    dim+1,
    3,
    "FESystem[FE_Q(2)^d-FE_Q(1)]",
    (dim==3)?"u,u,u,p":"u,u,p",
    "1,0"),
  AMG_A("AMG for A"),
  AMG_Ap("AMG for Ap"),
  jac_Mp("Jacobi for Mp", 1.4),
  dynamic(dynamic),
  compute_Mp(false),
  compute_Ap(false),
  is_parallel(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
{
  this->init();
}

template <int dim, int spacedim, typename LAC>
void NavierStokes<dim,spacedim,LAC>::
declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, NavierStokes<dim,spacedim,LAC>,LAC>::
  declare_parameters(prm);
  this->add_parameter(prm, &prec_name,  "Preconditioner","default",
                      Patterns::Selection("default|identity|low-nu|cah-cha"),
                      "Available preconditioners: \n"
                      " - default  -> S^-1 = nu * Mp^-1 \n"
                      " - identity -> S^-1 = identity \n"
                      " - low-nu   -> S^-1 = rho * alpha * Ap^-1 \n"
                      " - cah-cha  -> S^-1 = nu * Mp^-1 + rho * alpha * Ap^-1 ");
  this->add_parameter(prm, &compute_force,
                      "Compute sigma on the obstacle", "false",
                      Patterns::Bool(),
                      "Compute the mean resulting force acting\n"
                      "on the obstacle.");
  this->add_parameter(prm, &SUPG_alpha,
                      "SUPG alpha", "0.0",
                      Patterns::Double(0.0),
                      "Use SUPG alpha");
  this->add_parameter(prm, &use_skew_symmetric_advection,
                      "Use Skew symmetric form", "true",
                      Patterns::Bool(),
                      "");
  this->add_parameter(prm, &dynamic,
                      "Enable dynamic term (\\partial_t u)", "true",
                      Patterns::Bool(),
                      "Enable the dynamic term of the equation.");
  this->add_parameter(prm, &non_linear_term, "Non linear term","linear",
                      Patterns::Selection("fully_non_linear|linear|RHS"),
                      "Available options: \n"
                      " fully_non_linear\n"
                      " linear\n"
                      " RHS\n");
  this->add_parameter(prm, &linearize_in_time,
                      "Linearize using time", "true",
                      Patterns::Bool(),
                      "If true use the solution of the previous time step\n"
                      "to linearize the non-linear term, otherwise use the\n" "solution of the previous step (of an iterative methos).");
  this->add_parameter(prm, &gamma,
                      "div-grad stabilization parameter", "0.0",
                      Patterns::Double(0.0),
                      "");
  this->add_parameter(prm, &gamma_p,
                      "p-q stabilization parameter", "0.0",
                      Patterns::Double(0.0),
                      "");
  this->add_parameter(prm, &rho,
                      "rho [kg m^3]", "1.0",
                      Patterns::Double(0.0),
                      "Density");
  this->add_parameter(prm, &nu,
                      "nu [Pa s]", "1.0",
                      Patterns::Double(0.0),
                      "Viscosity");
  this->add_parameter(prm, &invert_Ap,
                      "Invert Ap using inverse_operator", "true", Patterns::Bool());
  this->add_parameter(prm, &invert_Mp,
                      "Invert Mp using inverse_operator", "true", Patterns::Bool());
  this->add_parameter(prm, &CG_solver_tolerance,
                      "CG Solver tolerance", "1e-8",
                      Patterns::Double(0.0));
  this->add_parameter(prm, &GMRES_solver_tolerance,
                      "GMRES Solver tolerance", "1e-8",
                      Patterns::Double(0.0));
}

template <int dim, int spacedim, typename LAC>
void
NavierStokes<dim,spacedim,LAC>::
solution_preprocessing(FEValuesCache<dim,spacedim> &fe_cache) const
{
  if (!compute_force) return;

  Tensor<1, dim, double> global_force;

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;

  auto local_copy = [this, &global_force]
                    (const NSUtilities::CopyForce<dim> &data)
  {
    global_force += data.local_force;
  };

  auto local_assemble = [this]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &fe_cache,
                         NSUtilities::CopyForce<dim> &data)
  {
    if (compute_force && is_parallel)
      {
        double dummy = 0.0;

        const FEValuesExtractors::Vector velocity(0);
        const FEValuesExtractors::Scalar pressure(dim);


        for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
          {
            unsigned int face_id = cell->face(face)->boundary_id();
            if (cell->face(face)->at_boundary() && ((face_id == 5 ) || (face_id ==6)) )
              {
                this->reinit(dummy, cell, face, fe_cache);
// Velocity:
                auto &sym_grad_u_ = fe_cache.get_symmetric_gradients( "explicit_solution", "f_grad_u", velocity, dummy);
                auto &p_ = fe_cache.get_values( "explicit_solution", "f_p", pressure, dummy);

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
                    data.local_force -= force; // Minus is due to normal issue..

                  } // end loop over quadrature points
                break;
              } // endif face->at_boundary
          } // end loop over faces
      }
  };


  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   this->get_dof_handler().begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   this->get_dof_handler().end()),
       local_assemble,
       local_copy,
       fe_cache,
       NSUtilities::CopyForce<dim>());

  auto &cache = fe_cache.get_cache();

  global_force[0] = Utilities::MPI::sum(global_force[0],MPI_COMM_WORLD);
  global_force[1] = Utilities::MPI::sum(global_force[1],MPI_COMM_WORLD);

  cache.template add_copy<Tensor<1, dim, double> >(global_force, "global_force");

  output_force = global_force;

  return;
}

template<int dim, int spacedim, typename LAC>
void
NavierStokes<dim,spacedim,LAC>::
output_solution (const unsigned int &current_cycle,
                 const unsigned int &step_number) const
{
  std::stringstream suffix;
  suffix << "." << current_cycle << "." << step_number;
  this->data_out.prepare_data_output( this->get_dof_handler(),
                                      suffix.str());
  this->data_out.add_data_vector (this->get_locally_relevant_solution(), this->get_component_names());
  std::vector<std::string> sol_dot_names =
    Utilities::split_string_list(this->get_component_names());
  for (auto &name : sol_dot_names)
    {
      name += "_dot";
    }
  this->data_out.add_data_vector (this->get_locally_relevant_solution_dot(), print(sol_dot_names, ","));

  this->data_out.write_data_and_clear(this->get_output_mapping());

  auto &pcout = this->get_pcout();
  if (compute_force && is_parallel)
    pcout << " Total force on the sphere (vertical value): "  << std::endl
          << "     f_x = " << output_force[0] << std::endl
          << "     f_y = " << output_force[1] << std::endl << std::endl
          << "============================================================="
          << std::endl << std::endl << std::endl;
}


template <int dim, int spacedim, typename LAC>
void NavierStokes<dim,spacedim,LAC>::
parse_parameters_call_back ()
{
  if (prec_name == "default")
    compute_Mp = true;
  else if (prec_name == "low-nu")
    compute_Ap = true;
  else if (prec_name == "cah-cha")
    {
      compute_Mp = true;
      compute_Ap = true;
    }

  // p-q stabilization term:
  if (gamma_p!=0.0)
    compute_Mp = true;
}

template <int dim, int spacedim, typename LAC>
void NavierStokes<dim,spacedim,LAC>::
set_matrix_couplings(std::vector<std::string> &couplings) const
{
  if (is_parallel)
    couplings[0] = "1,1;0,0";
  else
    couplings[0] = "1,1;1,0";

  couplings[1] = "0,0;0,1";
  couplings[2] = "0,0;0,1";
}

template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
NavierStokes<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &residual,
                       bool compute_only_system_terms) const
{
  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);

  ResidualType et = 0;
  double dummy = 0.0;
  double h = cell->diameter();

  // dummy number to define the type of variables
  this->reinit (et, cell, fe_cache);

  // Velocity:
  auto &us = fe_cache.get_values("solution", "u", velocity, et);
  auto &div_us = fe_cache.get_divergences("solution", "div_u", velocity,et);
  auto &grad_us = fe_cache.get_gradients("solution", "grad_u", velocity,et);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients("solution", "sym_grad_u", velocity,et);
  auto &us_dot = fe_cache.get_values("solution_dot", "u_dot", velocity, et);

  // Velocity:
  auto &ps = fe_cache.get_values("solution", "p", pressure, et);
  auto &grad_ps = fe_cache.get_gradients("solution", "grad_p", pressure,et);

  // Previous time step solution:
  auto &u_olds = fe_cache.get_values("explicit_solution", "ue", velocity, dummy);
  auto &grad_u_olds = fe_cache.get_gradients("explicit_solution", "grad_ue", velocity, dummy);

  // Previous Jacobian step solution:
  fe_cache.cache_local_solution_vector("prev_solution",
                                       this->get_locally_relevant_solution(), dummy);
  auto &u_prevs = fe_cache.get_values("prev_solution", "up", velocity, dummy);
  auto &grad_u_prevs = fe_cache.get_gradients("prev_solution", "grad_up", velocity, dummy);

  const unsigned int n_quad_points = us.size();
  auto &JxW = fe_cache.get_JxW_values();

  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int quad=0; quad<n_quad_points; ++quad)
    {
      // Pressure:
      const ResidualType &p = ps[quad];
      const Tensor<1, dim, ResidualType> &grad_p = grad_ps[quad];

      // Velocity:
      const Tensor<1, dim, ResidualType> &u = us[quad];
      const Tensor<1, dim, ResidualType> &u_dot = us_dot[quad];
      const Tensor<2, dim, ResidualType> &grad_u = grad_us[quad];
      const Tensor<2, dim, ResidualType> &sym_grad_u = sym_grad_us[quad];
      const ResidualType &div_u = div_us[quad];

      // Previous time step solution:
      const Tensor<1, dim, ResidualType> &u_old = u_olds[quad];
      const Tensor<2, dim, ResidualType> &grad_u_old = grad_u_olds[quad];

      // Previous Jacobian step solution:
      const Tensor<1, dim, ResidualType> &u_prev = u_prevs[quad];
      const Tensor<2, dim, ResidualType> &grad_u_prev = grad_u_prevs[quad];

      for (unsigned int i=0; i<residual[0].size(); ++i)
        {
          // Velocity:
          auto v = fev[velocity ].value(i,quad);
          auto div_v = fev[velocity ].divergence(i,quad);
          auto sym_grad_v = fev[ velocity ].symmetric_gradient(i,quad);

          auto grad_v = fev[ velocity ].gradient(i,quad);
          // Pressure:
          auto q = fev[ pressure ].value(i,quad);
          auto grad_q = fev[ pressure ].gradient(i,quad);

          // Non-linear term:
          Tensor<1, dim, ResidualType> nl_u;
          ResidualType res = 0.0;

          // Time derivative:
          if (dynamic)
            res += rho * u_dot * v;

          Tensor<2, dim, ResidualType> gradoldu;
          Tensor<1, dim, ResidualType> oldu;

          if (linearize_in_time)
            {
              gradoldu=grad_u_old;
              oldu=u_old;
            }
          else
            {
              gradoldu=grad_u_prev;
              oldu=u_prev;
            }

          if (non_linear_term=="fully_non_linear")
            nl_u = grad_u * u;
          else if (non_linear_term=="linear")
            nl_u = grad_u * oldu;
          else if (non_linear_term=="RHS")
            nl_u = gradoldu * oldu;

          //    ResidualType non_linear_term = 0;

          if (use_skew_symmetric_advection)
            res += 0.5 * ( scalar_product( nl_u, v) + scalar_product(grad_v * oldu, u) );
          else
            res += scalar_product( nl_u, v);

          double norm = std::sqrt(SacadoTools::to_double(oldu*oldu));

          if (norm > 0 && SUPG_alpha > 0)
            res += scalar_product( nl_u, SUPG_alpha * (h/norm) * grad_v  * oldu);


          // grad-div stabilization term:
          if (gamma!=0.0)
            res += gamma * div_u * div_v;

          // Diffusion term:
          res += nu * scalar_product(sym_grad_u, sym_grad_v);

          // Pressure term:
          res -= p * div_v;

          // Incompressible constraint:
          // if serial I use a direct solver over the
          // whole (A B', B 0) and i need to assemble B
          // otherwise I use linear operators to implement
          // tha action of transpose(B').
          if (!is_parallel)
            res -= div_u * q;

          residual[0][i] += res * JxW[q];

          // Mp preconditioner:
          if (!compute_only_system_terms && compute_Mp)
            residual[1][i] += p * q * JxW[q];

          // Ap preconditioner:
          if (!compute_only_system_terms && compute_Ap)
            residual[2][i] += grad_p * grad_q * JxW[q];
        }
    }
  (void)compute_only_system_terms;
}


template <int dim, int spacedim, typename LAC>
void
NavierStokes<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
                                                         LinearOperator<LATrilinos::VectorType> &system_op,
                                                         LinearOperator<LATrilinos::VectorType> &prec_op,
                                                         LinearOperator<LATrilinos::VectorType> &prec_op_finer) const
{

  auto alpha = this->get_alpha();
  // double dt = this->get_timestep();

  typedef LATrilinos::VectorType::BlockType  BVEC;
  typedef LATrilinos::VectorType             VEC;

  static ReductionControl solver_control_cg(matrices[0]->m(), CG_solver_tolerance);
  static SolverCG<BVEC> solver_CG(solver_control_cg);

  static ReductionControl solver_control_gmres(matrices[0]->m(), GMRES_solver_tolerance);
  static SolverGMRES<BVEC> solver_GMRES(solver_control_gmres);


  // SYSTEM MATRIX:
  auto A = linear_operator<BVEC>( matrices[0]->block(0,0) );
  auto Bt = linear_operator<BVEC>( matrices[0]->block(0,1) );
  auto B = transpose_operator<BVEC>( Bt );
  auto C  = B*Bt;
  auto ZeroP = null_operator(C);

  if (gamma_p!=0.0)
    C = linear_operator<BVEC>( matrices[1]->block(1,1) );
  else
    C = ZeroP;

  // ASSEMBLE THE PROBLEM:
  system_op = block_operator<2, 2, VEC>({{
      {{ A, Bt }} ,
      {{ B, C }}
    }
  });

  const DoFHandler<dim,spacedim> &dh = this->get_dof_handler();
  const ParsedFiniteElement<dim,spacedim> &fe = this->pfe;

  AMG_A.initialize_preconditioner<dim, spacedim>( matrices[0]->block(0,0), fe, dh);

  auto A_inv = linear_operator<BVEC>(matrices[0]->block(0,0), AMG_A);
  auto A_inv_finer = inverse_operator(A, solver_GMRES, AMG_A);

  LinearOperator<BVEC> Schur_inv;
  LinearOperator<BVEC> Ap, Ap_inv, Mp, Mp_inv;

  if (compute_Mp)
    {
      Mp = linear_operator<BVEC>( matrices[1]->block(1,1) );
      jac_Mp.initialize_preconditioner<>(matrices[1]->block(1,1));
      Mp_inv = inverse_operator( Mp, solver_CG, jac_Mp);
    }
  if (compute_Ap)
    {
      Ap = linear_operator<BVEC>( matrices[2]->block(1,1) );
      AMG_Ap.initialize_preconditioner<dim, spacedim>(matrices[2]->block(1,1), fe, dh);

      if (invert_Ap)
        {
          Ap_inv  = inverse_operator( Ap, solver_CG, AMG_Ap);
        }
      else
        {
          Ap_inv = linear_operator<BVEC>(matrices[2]->block(1,1), AMG_Ap);
        }
    }

  LinearOperator<BVEC> P00,P01,P10,P11;
  LinearOperator<BVEC> P00_finer,P01_finer,P10_finer,P11_finer;

  if (prec_name=="default" || prec_name=="cah-cha")
    Schur_inv = (gamma + 1/nu) * Mp_inv;
  else if (prec_name=="low-nu" || prec_name=="cah-cha")
    Schur_inv += (alpha * rho)  * Ap_inv;
  else if (prec_name=="identity")
    Schur_inv = identity_operator((C).reinit_range_vector);


  BlockLinearOperator<VEC> M = block_operator<2, 2, VEC>({{
      {{ null_operator(A), Bt               }},
      {{ null_operator(B), null_operator(C) }}
    }
  });

  // Preconditioner
  //////////////////////////////


  typedef LinearOperator<TrilinosWrappers::MPI::Vector,TrilinosWrappers::MPI::Vector> Op_MPI;

  BlockLinearOperator<VEC> diag_inv
  = block_diagonal_operator<2, VEC>(std::array<Op_MPI,2>({{ A_inv, -1 * Schur_inv }}));
  prec_op = block_back_substitution(M, diag_inv);


  // Finer preconditioner
  //////////////////////////////
  BlockLinearOperator<VEC> diag_inv_finer
  = block_diagonal_operator<2, VEC>(std::array<Op_MPI,2>({{ A_inv_finer, -1 * Schur_inv }}));
  prec_op_finer = block_back_substitution(M, diag_inv_finer);
}

#endif

/*! @} */
