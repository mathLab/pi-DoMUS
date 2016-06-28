#include "pidomus.h"
#include "pidomus_macros.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


#include <deal.II/numerics/vector_tools.h>

#include <deal2lkit/utilities.h>
#include "lac/lac_initializer.h"

#include <typeinfo>
#include <limits>
#include <numeric>


using namespace dealii;
using namespace deal2lkit;


// This file contains the implementation of:
// - constructor
// - run()
// - make_grid_fe()
// - setup_dofs()
// - solve_jacobian_system()
//



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
      else if (time_stepper == "euler" || time_stepper == "imex")
        {
          current_alpha = euler.get_alpha();
          euler.start_ode(solution, solution_dot);
        }
      eh.error_from_exact(interface.get_error_mapping(), *dof_handler, locally_relevant_solution, exact_solution);
    }

  eh.output_table(pcout);

}


template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::make_grid_fe()
{
  auto _timer = computing_timer.scoped_timer("Make grid and finite element");
  signals.begin_make_grid_fe();
  triangulation = SP(pgg.distributed(comm));
  dof_handler = SP(new DoFHandler<dim, spacedim>(*triangulation));
  signals.postprocess_newly_created_triangulation(*triangulation);
  fe = SP(interface.pfe());
  triangulation->refine_global (initial_global_refinement);
  signals.end_make_grid_fe();
}



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
          VectorTools::interpolate(interface.get_interpolate_mapping(), *dof_handler, initial_solution, solution);
          VectorTools::interpolate(interface.get_interpolate_mapping(), *dof_handler, initial_solution_dot, solution_dot);
        }
      else if (!we_are_parallel)
        {
          const QGauss<dim> quadrature_formula(fe->degree + 1);
          VectorTools::project(interface.get_project_mapping(), *dof_handler, constraints, quadrature_formula, initial_solution, solution);
          VectorTools::project(interface.get_project_mapping(), *dof_handler, constraints, quadrature_formula, initial_solution_dot, solution_dot);
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



#define INSTANTIATE(dim,spacedim,LAC) \
  template class piDoMUS<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)


