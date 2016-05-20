#include "pidomus.h"
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/identity_matrix.h>
//
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#ifdef DEAL_II_WITH_ARPACK
#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/parpack_solver.h>
#include <deal.II/lac/iterative_inverse.h>
#endif

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/distributed/solution_transfer.h>
//#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal2lkit/utilities.h>

#include <typeinfo>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <numeric>
#include <locale>
#include <string>
#include <math.h>
#include <cmath>

#include <Teuchos_TimeMonitor.hpp>

#include "lac/lac_initializer.h"

using namespace dealii;
using namespace deal2lkit;


/* ------------------------ CONSTRUCTORS ------------------------ */

template <int dim, int spacedim, typename LAC>
piDoMUS<dim, spacedim, LAC>::piDoMUS (const std::string &name,
                                      const BaseInterface<dim, spacedim, LAC> &interface,
                                      const MPI_Comm &communicator)
  :
  ParameterAcceptor(name),
  SundialsInterface<typename LAC::VectorType>(communicator),
  comm(communicator),
  interface(interface),
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(comm)
          == 0)),

  pgg("Domain"),

  pgr("Refinement"),


  current_time(std::numeric_limits<double>::quiet_NaN()),
  // IDA calls residual first with alpha = 0
  current_alpha(0.0),
  current_dt(std::numeric_limits<double>::quiet_NaN()),
  previous_time(std::numeric_limits<double>::quiet_NaN()),
  previous_dt(std::numeric_limits<double>::quiet_NaN()),
  second_to_last_time(std::numeric_limits<double>::quiet_NaN()),
  second_to_last_dt(std::numeric_limits<double>::quiet_NaN()),

  n_matrices(interface.n_matrices),

  eh("Error Tables", interface.get_component_names(),
     print(std::vector<std::string>(interface.n_components, "L2,H1"), ";")),

  exact_solution("Exact solution",
                 interface.n_components),
  initial_solution("Initial solution",
                   interface.n_components),
  initial_solution_dot("Initial solution_dot",
                       interface.n_components),

  forcing_terms("Forcing terms",
                interface.n_components,
                interface.get_component_names(), ""),
  neumann_bcs("Neumann boundary conditions",
              interface.n_components,
              interface.get_component_names(), ""),
  dirichlet_bcs("Dirichlet boundary conditions",
                interface.n_components,
                interface.get_component_names(), "0=ALL"),
  dirichlet_bcs_dot("Time derivative of Dirichlet boundary conditions",
                    interface.n_components,
                    interface.get_component_names(), ""),

  zero_average("Zero average constraints",
               interface.n_components,
               interface.get_component_names() ),


  ida(*this),
  euler(*this),
  we_are_parallel(Utilities::MPI::n_mpi_processes(comm) > 1)
{

  interface.initialize_simulator (*this);


  for (unsigned int i=0; i<n_matrices; ++i)
    {
      matrices.push_back( SP( new typename LAC::BlockMatrix() ) );
      matrix_sparsities.push_back( SP( new typename LAC::BlockSparsityPattern() ) );
    }

}



/* ------------------------ PARAMETERS ------------------------ */

template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::
declare_parameters (ParameterHandler &prm)
{
  add_parameter(  prm,
                  &initial_global_refinement,
                  "Initial global refinement",
                  "1",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &n_cycles,
                  "Number of cycles",
                  "1",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &max_time_iterations,
                  "Maximum number of time steps",
                  "10000",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &jacobian_solver_tolerance,
                  "Jacobian solver tolerance",
                  "1e-8",
                  Patterns::Double (0.0));

  add_parameter(  prm,
                  &adaptive_refinement,
                  "Adaptive refinement",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &use_direct_solver,
                  "Use direct solver if available",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &verbose,
                  "Print some useful informations about processes",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &overwrite_iter,
                  "Overwrite Newton's iterations",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &time_stepper,
                  "Time stepper",
                  "euler",
                  Patterns::Selection("ida|euler"));

  add_parameter(  prm,
                  &use_space_adaptivity,
                  "Refine mesh during transient",
                  "false",
                  Patterns::Bool());

  add_parameter(  prm,
                  &kelly_threshold,
                  "Threshold for solver's restart",
                  "1e-2",
                  Patterns::Double(0.0));

  add_parameter(  prm,
                  &output_timer,
                  "Show timer",
                  "false",
                  Patterns::Bool());

  add_parameter(  prm,
                  &max_iterations,
                  "Max iterations",
                  "50",
                  Patterns::Integer (0),
                  "Maximum number of iterations for solving the Newtons's system.\n"
                  "If this variables is 0, then the size of the matrix is used.");

  add_parameter(  prm,
                  &max_iterations_finer,
                  "Max iterations finer prec.",
                  "0",
                  Patterns::Integer (0),
                  "Maximum number of iterations for solving the Newtons's system \n"
                  "using the finer preconditioner.\n"
                  "If this variables is 0, then the size of the matrix is used.");

  add_parameter(  prm,
                  &enable_finer_preconditioner,
                  "Enable finer preconditioner",
                  "false",
                  Patterns::Bool());

  add_parameter(  prm,
                  &max_tmp_vector,
                  "Max tmp vectors",
                  "30",
                  Patterns::Integer (1),
                  "Maximum number of temporary vectors used by FGMRES for the \n"
                  "solution of the linear system using the coarse preconditioner.");

  add_parameter(  prm,
                  &max_tmp_vector_finer,
                  "Max tmp vectors for finer system",
                  "50",
                  Patterns::Integer (1),
                  "Maximum number of temporary vectors used by FGMRES for the \n"
                  "solution of the linear system using the finer preconditioner.");

#ifdef DEAL_II_WITH_ARPACK
  add_parameter(  prm,
                  &n_eigenvalues,
                  "Number of eigenvalues to compute",
                  "10",
                  Patterns::Integer (1));

  add_parameter(  prm,
                  &n_arnoldi_vectors,
                  "Number of used Arnoldi vectors",
                  "0",
                  Patterns::Integer (0),
                  "If 0, the number of vectors used will be\n"
                  "2*number_of_eigenvalues+2");

  add_parameter(  prm,
                  &which_eigenvalues,
                  "Which eigenvalues",
                  "smallest_real_part",
                  Patterns::Selection("algebraically_largest"
                                      "|algebraically_smallest"
                                      "|largest_magnitude"
                                      "|smallest_magnitude"
                                      "|largest_real_part"
                                      "|smallest_real_part"
                                      "|largest_imaginary_part"
                                      "|smallest_imaginary_part"
                                      "|both_ends"));
#endif
}




template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::
apply_neumann_bcs (
  const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
  FEValuesCache<dim,spacedim> &scratch,
  std::vector<double> &local_residual) const
{

  double dummy = 0.0;

  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      unsigned int face_id = cell->face(face)->boundary_id();
      if (cell->face(face)->at_boundary() && neumann_bcs.acts_on_id(face_id))
        {
          interface.reinit(dummy, cell, face, scratch);

          auto &fev = scratch.get_current_fe_values();
          auto &q_points = scratch.get_quadrature_points();
          auto &JxW = scratch.get_JxW_values();

          for (unsigned int q=0; q<q_points.size(); ++q)
            {
              Vector<double> T(interface.n_components);
              neumann_bcs.get_mapped_function(face_id)->vector_value(q_points[q], T);

              for (unsigned int i=0; i<local_residual.size(); ++i)
                for (unsigned int c=0; c<interface.n_components; ++c)
                  local_residual[i] -= T[c]*fev.shape_value_component(i,q,c)*JxW[q];

            }// end loop over quadrature points

          break;

        } // endif face->at_boundary

    }// end loop over faces

}// end function definition



template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::
apply_forcing_terms (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                     FEValuesCache<dim,spacedim> &scratch,
                     std::vector<double> &local_residual) const
{
  unsigned cell_id = cell->material_id();
  if (forcing_terms.acts_on_id(cell_id))
    {
      double dummy = 0.0;
      interface.reinit(dummy, cell, scratch);

      auto &fev = scratch.get_current_fe_values();
      auto &q_points = scratch.get_quadrature_points();
      auto &JxW = scratch.get_JxW_values();
      for (unsigned int q=0; q<q_points.size(); ++q)
        for (unsigned int i=0; i<local_residual.size(); ++i)
          for (unsigned int c=0; c<interface.n_components; ++c)
            {
              double B = forcing_terms.get_mapped_function(cell_id)->value(q_points[q],c);
              local_residual[i] -= B*fev.shape_value_component(i,q,c)*JxW[q];
            }
    }
}


template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::
apply_dirichlet_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                     const ParsedDirichletBCs<dim,spacedim> &bc,
                     ConstraintMatrix &constraints) const
{
  try
    {
      bc.interpolate_boundary_values(interface.get_mapping(),dof_handler,constraints);
    }
  catch (...)
    {
      AssertThrow(!we_are_parallel,
                  ExcMessage("You called VectorTools::project_boundary_values(), which is not \n"
                             "currently supported on deal.II in parallel settings.\n"
                             "Feel free to submit a patch :)"));
      const QGauss<dim-1> quad(fe->degree+1);
      bc.project_boundary_values(interface.get_mapping(),dof_handler,quad,constraints);
    }
  unsigned int codim = spacedim - dim;
  if (codim == 0)
    bc.compute_nonzero_normal_flux_constraints(dof_handler,interface.get_mapping(),constraints);
}



template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::parse_parameters_call_back()
{
  use_direct_solver &= (typeid(typename LAC::BlockMatrix) == typeid(dealii::BlockSparseMatrix<double>));
#ifdef DEAL_II_WITH_ARPACK
  if (n_arnoldi_vectors == 0)
    n_arnoldi_vectors = 2*n_eigenvalues+2;
#endif
}


/* ------------------------ DEGREE OF FREEDOM ------------------------ */

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::setup_dofs (const bool &first_run)
{
  auto _timer = computing_timer.scoped_timer("Setup dof systems");
  signals.begin_setup_dofs();
  std::vector<unsigned int> sub_blocks = interface.pfe.get_component_blocks();
  dof_handler->distribute_dofs (*fe);
  DoFRenumbering::component_wise (*dof_handler, sub_blocks);

  dofs_per_block.clear();
  dofs_per_block.resize(interface.pfe.n_blocks());

  DoFTools::count_dofs_per_block (*dof_handler, dofs_per_block,
                                  sub_blocks);

  std::locale s = pcout.get_stream().getloc();
  pcout.get_stream().imbue(std::locale(""));
  pcout << "Number of active cells: "
        << triangulation->n_global_active_cells()
        << " (on "
        << triangulation->n_levels()
        << " levels)"
        << std::endl
        << "Number of degrees of freedom: "
        << dof_handler->n_dofs()
        << "(" << print(dofs_per_block, "+") << ")"
        << std::endl
        << std::endl;
  pcout.get_stream().imbue(s);


  partitioning.resize(0);
  relevant_partitioning.resize(0);

  IndexSet relevant_set;
  {
    global_partitioning = dof_handler->locally_owned_dofs();
    for (unsigned int i = 0; i < interface.pfe.n_blocks(); ++i)
      partitioning.push_back(global_partitioning.get_view( std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i, 0),
                                                           std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i + 1, 0)));

    DoFTools::extract_locally_relevant_dofs (*dof_handler,
                                             relevant_set);

    for (unsigned int i = 0; i < interface.pfe.n_blocks(); ++i)
      relevant_partitioning.push_back(relevant_set.get_view(std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i, 0),
                                                            std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i + 1, 0)));
  }

  update_functions_and_constraints(current_time);

  ScopedLACInitializer initializer(dofs_per_block,
                                   partitioning,
                                   relevant_partitioning,
                                   comm);

  initializer(solution);
  initializer(solution_dot);


  if (we_are_parallel)
    {
      initializer.ghosted(locally_relevant_solution);
      initializer.ghosted(locally_relevant_solution_dot);
      initializer.ghosted(locally_relevant_explicit_solution);
      initializer.ghosted(locally_relevant_previous_explicit_solution);
    }
  else
    {
      initializer(locally_relevant_solution);
      initializer(locally_relevant_solution_dot);
      initializer(locally_relevant_explicit_solution);
      initializer(locally_relevant_previous_explicit_solution);
    }


  for (unsigned int i=0; i < n_matrices; ++i)
    {
      matrices[i]->clear();
      initializer(*matrix_sparsities[i],
                  *dof_handler,
                  constraints,
                  interface.get_matrix_coupling(i));
      matrices[i]->reinit(*matrix_sparsities[i]);
    }

  if (first_run)
    {
      if (fe->has_support_points())
        {
          VectorTools::interpolate(interface.get_mapping(), *dof_handler, initial_solution, solution);
          VectorTools::interpolate(interface.get_mapping(), *dof_handler, initial_solution_dot, solution_dot);
        }
      else if (!we_are_parallel)
        {
          const QGauss<dim> quadrature_formula(fe->degree + 1);
          VectorTools::project(interface.get_mapping(), *dof_handler, constraints, quadrature_formula, initial_solution, solution);
          VectorTools::project(interface.get_mapping(), *dof_handler, constraints, quadrature_formula, initial_solution_dot, solution_dot);
        }
      else
        {
          Point<spacedim> p;
          Vector<double> vals(interface.n_components);
          Vector<double> vals_dot(interface.n_components);
          initial_solution.vector_value(p, vals);
          initial_solution_dot.vector_value(p, vals_dot);

          unsigned int comp = 0;
          for (unsigned int b=0; b<solution.n_blocks(); ++b)
            {
              solution.block(b) = vals[comp];
              solution_dot.block(b) = vals_dot[comp];
              comp += fe->element_multiplicity(b);
            }
        }

      signals.fix_initial_conditions(solution, solution_dot);
      locally_relevant_explicit_solution = solution;

    }
  signals.end_setup_dofs();
}


template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::update_functions_and_constraints (const double &t)
{
  if (!std::isnan(t))
    {
      dirichlet_bcs.set_time(t);
      dirichlet_bcs_dot.set_time(t);
      forcing_terms.set_time(t);
      neumann_bcs.set_time(t);
    }
  constraints.clear();
  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints);

  apply_dirichlet_bcs(*dof_handler, dirichlet_bcs, constraints);

  zero_average.apply_zero_average_constraints(*dof_handler, constraints);

  constraints.close ();

  constraints_dot.clear();
  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints_dot);

  apply_dirichlet_bcs(*dof_handler, dirichlet_bcs_dot, constraints_dot);
  signals.update_constraint_matrices(constraints,constraints_dot);
  constraints_dot.close ();
}



template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::
syncronize(const double &t,
           const typename LAC::VectorType &solution,
           const typename LAC::VectorType &solution_dot)
{
  if (std::isnan(current_time))
    {
      // we are at the very first time step
      // solution = initial solution
      // explicit solution will be zero
      // previous explicit solution will be zero
      current_time = t;
      if (std::isnan(previous_time)) //very first step
        previous_time = t;
      update_functions_and_constraints(t);
      typename LAC::VectorType tmp(solution);
      typename LAC::VectorType tmp_dot(solution_dot);
      constraints.distribute(tmp);
      constraints_dot.distribute(tmp_dot);

      locally_relevant_solution = tmp;
      locally_relevant_solution_dot = tmp_dot;
      locally_relevant_explicit_solution = tmp;
    }
  else if (current_time < t) // next temporal step
    {
      second_to_last_dt = previous_dt;
      previous_dt       = current_dt;
      current_dt        = t - current_time;

      second_to_last_time = previous_time;
      previous_time = current_time;
      current_time = t;

      locally_relevant_previous_explicit_solution     = locally_relevant_explicit_solution;
      locally_relevant_explicit_solution     = solution;

      update_functions_and_constraints(t);
      typename LAC::VectorType tmp(solution);
      typename LAC::VectorType tmp_dot(solution_dot);
      constraints.distribute(tmp);
      constraints_dot.distribute(tmp_dot);

      locally_relevant_solution = tmp;
      locally_relevant_solution_dot = tmp_dot;
    }
  else if (current_time == t)
    {
      // we are calling this function in different part of the code
      // within the same time step (e.g. during the non-linear solver)
      // so no need to call udpdate_functions_and_constraints()

      typename LAC::VectorType tmp(solution);
      typename LAC::VectorType tmp_dot(solution_dot);
      constraints.distribute(tmp);
      constraints_dot.distribute(tmp_dot);
      locally_relevant_solution = tmp;
      locally_relevant_solution_dot = tmp_dot;
    }
  else
    {
      // we get here when t<current_time. This may happen when we are
      // trying to estimate the time step and we need to reduce the
      // first guess. IDA enters here often times
      current_time = t;
      update_functions_and_constraints(t);

      typename LAC::VectorType tmp(solution);
      typename LAC::VectorType tmp_dot(solution_dot);
      constraints.distribute(tmp);
      constraints_dot.distribute(tmp_dot);

      locally_relevant_solution = tmp;
      locally_relevant_solution_dot = tmp_dot;
    }

}



template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::assemble_matrices (const double t,
                                                     const typename LAC::VectorType &solution,
                                                     const typename LAC::VectorType &solution_dot,
                                                     const double alpha)
{
  auto _timer = computing_timer.scoped_timer ("Assemble matrices");

  signals.begin_assemble_matrices();

  current_alpha = alpha;
  syncronize(t,solution,solution_dot);

  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);

  FEValuesCache<dim,spacedim> fev_cache(interface.get_mapping(),
                                        *fe, quadrature_formula,
                                        interface.get_cell_update_flags(),
                                        face_quadrature_formula,
                                        interface.get_face_update_flags());

  interface.solution_preprocessing(fev_cache);

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;


  for (unsigned int i=0; i<n_matrices; ++i)
    *(matrices[i]) = 0;



  auto local_copy = [this]
                    (const pidomus::CopyData & data)
  {

    for (unsigned int i=0; i<n_matrices; ++i)
      this->constraints.distribute_local_to_global (data.local_matrices[i],
                                                    data.local_dof_indices,
                                                    *(this->matrices[i]));
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         pidomus::CopyData & data)
  {
    this->interface.assemble_local_matrices(cell, scratch, data);
  };



  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       fev_cache,
       pidomus::CopyData(fe->dofs_per_cell,n_matrices));

  for (unsigned int i=0; i<n_matrices; ++i)
    matrices[i]->compress(VectorOperation::add);
}


/* ------------------------ MESH AND GRID ------------------------ */

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::
refine_and_transfer_solutions(LATrilinos::VectorType &y,
                              LATrilinos::VectorType &y_dot,
                              LATrilinos::VectorType &locally_relevant_y,
                              LATrilinos::VectorType &locally_relevant_y_dot,
                              LATrilinos::VectorType &locally_relevant_y_expl,
                              bool adaptive_refinement)
{
  signals.begin_refine_and_transfer_solutions();
  locally_relevant_y = y;
  locally_relevant_y_dot = y_dot;

  parallel::distributed::SolutionTransfer<dim, LATrilinos::VectorType, DoFHandler<dim,spacedim> > sol_tr(*dof_handler);

  std::vector<const LATrilinos::VectorType *> old_sols (3);
  old_sols[0] = &locally_relevant_y;
  old_sols[1] = &locally_relevant_y_dot;
  old_sols[2] = &locally_relevant_y_expl;

  triangulation->prepare_coarsening_and_refinement();
  sol_tr.prepare_for_coarsening_and_refinement (old_sols);

  if (adaptive_refinement)
    triangulation->execute_coarsening_and_refinement ();
  else
    triangulation->refine_global (1);

  setup_dofs(false);


  LATrilinos::VectorType new_sol (y);
  LATrilinos::VectorType new_sol_dot (y);
  LATrilinos::VectorType new_sol_expl (y);

  std::vector<LATrilinos::VectorType *> new_sols (3);
  new_sols[0] = &new_sol;
  new_sols[1] = &new_sol_dot;
  new_sols[2] = &new_sol_expl;

  sol_tr.interpolate (new_sols);

  y = new_sol;
  y_dot = new_sol_dot;


  update_functions_and_constraints(current_time);
  constraints.distribute(y);
  constraints_dot.distribute(y_dot);

  locally_relevant_y = y;
  locally_relevant_y_dot = y_dot;
  locally_relevant_y_expl = new_sol_expl;

  signals.end_refine_and_transfer_solutions();
}

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::
refine_and_transfer_solutions(LADealII::VectorType &y,
                              LADealII::VectorType &y_dot,
                              LADealII::VectorType &locally_relevant_y,
                              LADealII::VectorType &locally_relevant_y_dot,
                              LADealII::VectorType &y_expl,
                              bool adaptive_refinement)
{
  signals.begin_refine_and_transfer_solutions();
  SolutionTransfer<dim, LADealII::VectorType, DoFHandler<dim,spacedim> > sol_tr(*dof_handler);

  std::vector<LADealII::VectorType> old_sols (3);
  old_sols[0] = y;
  old_sols[1] = y_dot;
  old_sols[2] = y_expl;

  triangulation->prepare_coarsening_and_refinement();
  sol_tr.prepare_for_coarsening_and_refinement (old_sols);

  if (adaptive_refinement)
    triangulation->execute_coarsening_and_refinement ();
  else
    triangulation->refine_global (1);

  setup_dofs(false);

  std::vector<LADealII::VectorType> new_sols (3);

  new_sols[0].reinit(y);
  new_sols[1].reinit(y_dot);
  new_sols[2].reinit(y_expl);

  sol_tr.interpolate (old_sols, new_sols);

  y      = new_sols[0];
  y_dot  = new_sols[1];
  y_expl = new_sols[2];

  update_functions_and_constraints(previous_time);
  constraints.distribute(y_expl);

  update_functions_and_constraints(current_time);
  constraints.distribute(y);
  constraints_dot.distribute(y_dot);

  locally_relevant_y = y;
  locally_relevant_y_dot = y_dot;

  signals.end_refine_and_transfer_solutions();
}

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::refine_mesh ()
{
  auto _timer = computing_timer.scoped_timer ("Mesh refinement");

  signals.begin_refine_mesh();

  if (adaptive_refinement)
    {
      Vector<float> estimated_error_per_cell (triangulation->n_active_cells());

      interface.estimate_error_per_cell(estimated_error_per_cell);

      pgr.mark_cells(estimated_error_per_cell, *triangulation);
    }

  refine_and_transfer_solutions(solution,
                                solution_dot,
                                locally_relevant_solution,
                                locally_relevant_solution_dot,
                                locally_relevant_explicit_solution,
                                adaptive_refinement);

  current_time = std::numeric_limits<double>::quiet_NaN();
  previous_time =   std::numeric_limits<double>::quiet_NaN();

  signals.end_refine_mesh();
}

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::make_grid_fe()
{
  signals.begin_make_grid_fe();
  triangulation = SP(pgg.distributed(comm));
  dof_handler = SP(new DoFHandler<dim, spacedim>(*triangulation));
  signals.postprocess_newly_created_triangulation(*triangulation);
  fe = SP(interface.pfe());
  triangulation->refine_global (initial_global_refinement);
  signals.end_make_grid_fe();
}

/* ------------------------ OUTPUTS ------------------------ */

template <int dim, int spacedim, typename LAC>
typename LAC::VectorType &
piDoMUS<dim, spacedim, LAC>::
get_solution()
{
  return solution;
}

/* ------------------------ RUN ------------------------ */

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::run ()
{
  interface.set_stepper(time_stepper);

  interface.connect_to_signals();

  for (current_cycle = 0; current_cycle < n_cycles; ++current_cycle)
    {
      if (current_cycle == 0)
        {
          make_grid_fe();
          setup_dofs(true);
        }
      else
        refine_mesh();

      constraints.distribute(solution);
      constraints_dot.distribute(solution_dot);

      if (time_stepper == "ida")
        ida.start_ode(solution, solution_dot, max_time_iterations);
      else if (time_stepper == "euler")
        {
          current_alpha = euler.get_alpha();
          euler.start_ode(solution, solution_dot);
        }
      eh.error_from_exact(interface.get_mapping(), *dof_handler, locally_relevant_solution, exact_solution);
    }

  eh.output_table(pcout);

}

/*** ODE Argument Interface ***/

template <int dim, int spacedim, typename LAC>
shared_ptr<typename LAC::VectorType>
piDoMUS<dim, spacedim, LAC>::create_new_vector() const
{
  shared_ptr<typename LAC::VectorType> ret = SP(new typename LAC::VectorType(solution));
  //   *ret *= 0;
  return ret;
}


template <int dim, int spacedim, typename LAC>
unsigned int
piDoMUS<dim, spacedim, LAC>::n_dofs() const
{
  return dof_handler->n_dofs();
}


template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::output_step(const double  t,
                                         const typename LAC::VectorType &solution,
                                         const typename LAC::VectorType &solution_dot,
                                         const unsigned int step_number,
                                         const double // h
                                        )
{
  auto _timer = computing_timer.scoped_timer ("Postprocessing");

  syncronize(t,solution,solution_dot);

  interface.output_solution(current_cycle,
                            step_number);
}



template <int dim, int spacedim, typename LAC>
bool
piDoMUS<dim, spacedim, LAC>::solver_should_restart(const double t,
                                                   const unsigned int /*step_number*/,
                                                   const double /*h*/,
                                                   typename LAC::VectorType &solution,
                                                   typename LAC::VectorType &solution_dot)
{

  auto _timer = computing_timer.scoped_timer ("Solver should restart");
  signals.begin_solver_should_restart();
  if (use_space_adaptivity)
    {
      double max_kelly=0;
      auto _timer = computing_timer.scoped_timer ("Compute error estimator");
      update_functions_and_constraints(t);

      constraints.distribute(solution);
      locally_relevant_solution = solution;
      constraints_dot.distribute(solution_dot);
      locally_relevant_solution_dot = solution_dot;

      Vector<float> estimated_error_per_cell (triangulation->n_active_cells());

      interface.estimate_error_per_cell(estimated_error_per_cell);

      max_kelly = estimated_error_per_cell.linfty_norm();
      max_kelly = Utilities::MPI::max(max_kelly, comm);

      if (max_kelly > kelly_threshold)

        {
          pcout << "  ################ restart ######### \n"
                << "max_kelly > threshold\n"
                << max_kelly  << " >  " << kelly_threshold
                << std::endl
                << "######################################\n";
          pgr.mark_cells(estimated_error_per_cell, *triangulation);

          refine_and_transfer_solutions(solution,
                                        solution_dot,
                                        locally_relevant_solution,
                                        locally_relevant_solution_dot,
                                        locally_relevant_explicit_solution,
                                        adaptive_refinement);

          update_functions_and_constraints(t);
          constraints.distribute(solution);
          constraints_dot.distribute(solution_dot);

          signals.fix_solutions_after_refinement(solution,solution_dot);

          MPI::COMM_WORLD.Barrier();
          current_time = std::numeric_limits<double>::quiet_NaN();
          signals.end_solver_should_restart();
          return true;
        }
      else // if max_kelly > kelly_threshold
        {
          signals.end_solver_should_restart();
          return false;
        }

    }
  else // use space adaptivity
    {
      signals.end_solver_should_restart();
      return false;
    }
}


template <int dim, int spacedim, typename LAC>
int
piDoMUS<dim, spacedim, LAC>::residual(const double t,
                                      const typename LAC::VectorType &solution,
                                      const typename LAC::VectorType &solution_dot,
                                      typename LAC::VectorType &dst)
{
  auto _timer = computing_timer.scoped_timer ("Residual");

  signals.begin_residual();

  syncronize(t,solution,solution_dot);

  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);


  FEValuesCache<dim,spacedim> fev_cache(interface.get_mapping(),
                                        *fe, quadrature_formula,
                                        interface.get_cell_update_flags(),
                                        face_quadrature_formula,
                                        interface.get_face_update_flags());

  interface.solution_preprocessing(fev_cache);

  dst = 0;

  auto local_copy = [&dst, this] (const pidomus::CopyData & data)
  {

    this->constraints.distribute_local_to_global (data.local_residual,
                                                  data.local_dof_indices,
                                                  dst);
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         pidomus::CopyData & data)
  {
    this->interface.assemble_local_system_residual(cell,scratch,data);
    // apply conservative loads
    this->apply_forcing_terms(cell, scratch, data.local_residual);

    if (cell->at_boundary())
      this->apply_neumann_bcs(cell, scratch, data.local_residual);


  };

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       fev_cache,
       pidomus::CopyData(fe->dofs_per_cell,n_matrices));


  dst.compress(VectorOperation::add);

  auto id = solution.locally_owned_elements();
  for (unsigned int i = 0; i < id.n_elements(); ++i)
    {
      auto j = id.nth_index_in_set(i);
      if (constraints.is_constrained(j))
        dst[j] = solution(j) - locally_relevant_solution(j);
    }

  dst.compress(VectorOperation::insert);

  signals.end_residual();

  return 0;
}


template <int dim, int spacedim, typename LAC>
int
piDoMUS<dim, spacedim, LAC>::solve_jacobian_system(const double /*t*/,
                                                   const typename LAC::VectorType &/*y*/,
                                                   const typename LAC::VectorType &/*y_dot*/,
                                                   const typename LAC::VectorType &,
                                                   const double /*alpha*/,
                                                   const typename LAC::VectorType &src,
                                                   typename LAC::VectorType &dst) const
{
  auto _timer = computing_timer.scoped_timer ("Solve system");

  signals.begin_solve_jacobian_system();

  set_constrained_dofs_to_zero(dst);

  typedef dealii::BlockSparseMatrix<double> sMAT;
  typedef dealii::BlockVector<double> sVEC;

  if (we_are_parallel == false &&
      use_direct_solver == true)
    {

      SparseDirectUMFPACK inverse;
      inverse.factorize((sMAT &) *matrices[0]);
      inverse.vmult((sVEC &)dst, (sVEC &)src);

    }
  else
    {
      unsigned int tot_iteration = 0;
      try
        {
          unsigned int solver_iterations = matrices[0]->m();
          if ( max_iterations != 0 )
            solver_iterations = max_iterations;

          if (enable_finer_preconditioner && verbose)
            pcout << " --> Coarse preconditioner "
                  << std::endl;

          PrimitiveVectorMemory<typename LAC::VectorType> mem;

          SolverControl solver_control (solver_iterations,
                                        jacobian_solver_tolerance);

          SolverFGMRES<typename LAC::VectorType>
          solver(solver_control, mem,
                 typename SolverFGMRES<typename LAC::VectorType>::AdditionalData(max_tmp_vector, true));

          auto S_inv = inverse_operator(jacobian_op, solver, jacobian_preconditioner_op);
          S_inv.vmult(dst, src);

          tot_iteration += solver_control.last_step();
        }
      catch (const std::exception &e)
        {

          if (enable_finer_preconditioner)
            {
              unsigned int solver_iterations = matrices[0]->m();
              if ( max_iterations_finer != 0 )
                solver_iterations = max_iterations_finer;

              if (verbose)
                pcout << " --> Finer preconditioner "
                      << std::endl;

              PrimitiveVectorMemory<typename LAC::VectorType> mem;
              SolverControl solver_control (solver_iterations,
                                            jacobian_solver_tolerance);

              SolverFGMRES<typename LAC::VectorType>
              solver(solver_control, mem,
                     typename SolverFGMRES<typename LAC::VectorType>::AdditionalData(max_tmp_vector_finer, true));

              auto S_inv = inverse_operator(jacobian_op, solver, jacobian_preconditioner_op_finer);
              S_inv.vmult(dst, src);

              tot_iteration += solver_control.last_step();
            }
          else
            {
              AssertThrow(false,ExcMessage(e.what()));
            }

        }

      if (verbose)
        {
          if (!overwrite_iter)
            pcout << std::endl;
          pcout << " iterations:            "
                << tot_iteration;
          if (overwrite_iter)
            pcout << "               \r";
          else
            pcout << "               " << std::endl;
        }

    }

  set_constrained_dofs_to_zero(dst);
  signals.end_solve_jacobian_system();
  return 0;
}

template <int dim, int spacedim, typename LAC>
int
piDoMUS<dim,spacedim,LAC>::jacobian_vmult(const typename LAC::VectorType &src, typename LAC::VectorType &dst) const
{
  auto _timer = computing_timer.scoped_timer ("Jacobian vmult");

  if (we_are_parallel == false &&
      use_direct_solver == true)
    matrices[0]->vmult(dst, src);
  else
    jacobian_op.vmult(dst, src);

  return 0;
}

template <int dim, int spacedim, typename LAC>
int
piDoMUS<dim, spacedim, LAC>::setup_jacobian(const double t,
                                            const typename LAC::VectorType &src_yy,
                                            const typename LAC::VectorType &src_yp,
                                            const typename LAC::VectorType &,
                                            const double alpha)
{
  auto _timer = computing_timer.scoped_timer ("Setup Jacobian");

  signals.begin_setup_jacobian();

  current_alpha = alpha;
  syncronize(t,solution,solution_dot);

  assemble_matrices(t, src_yy, src_yp, alpha);
  if (use_direct_solver == false)
    {
      auto _timer = computing_timer.scoped_timer ("Compute system operators");

      interface.compute_system_operators(matrices,
                                         jacobian_op,
                                         jacobian_preconditioner_op,
                                         jacobian_preconditioner_op_finer);
    }
  signals.end_setup_jacobian();
  return 0;
}



template <int dim, int spacedim, typename LAC>
typename LAC::VectorType &
piDoMUS<dim, spacedim, LAC>::differential_components() const
{
  static typename LAC::VectorType diff_comps;
  diff_comps.reinit(solution);
  std::vector<unsigned int> block_diff = interface.get_differential_blocks();
  for (unsigned int i = 0; i < block_diff.size(); ++i)
    diff_comps.block(i) = block_diff[i];
  signals.fix_differential_components(diff_comps);
  set_constrained_dofs_to_zero(diff_comps);
  return diff_comps;
}



template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::set_constrained_dofs_to_zero(typename LAC::VectorType &v) const
{
  if (global_partitioning.n_elements() > 0)
    {
      auto k = global_partitioning.nth_index_in_set(0);
      if (constraints.is_constrained(k))
        v[k] = 0;
      else
        v[k] = v[k];
      for (unsigned int i = 1; i < global_partitioning.n_elements(); ++i)
        {
          auto j = global_partitioning.nth_index_in_set(i);
          if (constraints.is_constrained(j))
            v[j] = 0;
        }
      v.compress(VectorOperation::insert);
    }
}


template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::get_lumped_mass_matrix(typename LAC::VectorType &dst) const
{
  auto _timer = computing_timer.scoped_timer ("Assemble lumped mass matrix");


  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);


  FEValuesCache<dim,spacedim> fev_cache(interface.get_mapping(),
                                        *fe, quadrature_formula,
                                        interface.get_cell_update_flags(),
                                        face_quadrature_formula,
                                        interface.get_face_update_flags());


  dst = 0;

  auto local_copy = [&dst, this] (const pidomus::CopyData & data)
  {

    for (unsigned int i=0; i<data.local_dof_indices.size(); ++i)
      {
        // kinsol needs that each component must strictly greater
        // than zero.
        dst[ data.local_dof_indices[i] ] += data.local_residual[i] + 1e-12;
      }
  };

  auto local_assemble = []
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         pidomus::CopyData & data)
  {
    const unsigned dofs_per_cell = data.local_residual.size();
    scratch.reinit(cell);

    cell->get_dof_indices (data.local_dof_indices);

    auto &JxW = scratch.get_JxW_values();
    auto &fev = scratch.get_current_fe_values();

    const unsigned int n_q_points = JxW.size();

    std::fill(data.local_residual.begin(), data.local_residual.end(), 0.0);
    for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        data.local_residual[i] += fev.shape_value(i,q)*JxW[q];

  };

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       fev_cache,
       pidomus::CopyData(fe->dofs_per_cell,n_matrices));


  dst.compress(VectorOperation::add);

}


template class piDoMUS<2, 2, LATrilinos>;
template class piDoMUS<2, 3, LATrilinos>;
template class piDoMUS<3, 3, LATrilinos>;

template class piDoMUS<2, 2, LADealII>;
template class piDoMUS<2, 3, LADealII>;
template class piDoMUS<3, 3, LADealII>;

