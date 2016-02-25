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
//
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/linear_operator.h>

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

#include <Teuchos_TimeMonitor.hpp>

#include "lac/lac_initializer.h"

using namespace dealii;
using namespace deal2lkit;

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
}


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

  we_are_parallel(Utilities::MPI::n_mpi_processes(comm) > 1),

  old_t(-std::numeric_limits<double>::max())
{
  for (unsigned int i=0; i<n_matrices; ++i)
    {
      matrices.push_back( SP( new typename LAC::BlockMatrix() ) );
      matrix_sparsities.push_back( SP( new typename LAC::BlockSparsityPattern() ) );
    }
}


/* ------------------------ DEGREE OF FREEDOM ------------------------ */

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::setup_dofs (const bool &first_run)
{
  auto _timer = computing_timer.scoped_timer("Setup dof systems");
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
  constraints.clear ();
  constraints.reinit (relevant_set);

  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints);

  apply_dirichlet_bcs(*dof_handler, dirichlet_bcs, constraints);
  zero_average.apply_zero_average_constraints(*dof_handler, constraints);
  constraints.close ();

  constraints_dot.clear();
  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints_dot);

  apply_dirichlet_bcs(*dof_handler, dirichlet_bcs_dot, constraints_dot);

  constraints_dot.close ();

  ScopedLACInitializer initializer(dofs_per_block,
                                   partitioning,
                                   relevant_partitioning,
                                   comm);

  initializer(solution);
  initializer(solution_dot);
  initializer(explicit_solution);
  if (we_are_parallel)
    {
      initializer.ghosted(distributed_solution);
      initializer.ghosted(distributed_solution_dot);
      initializer.ghosted(distributed_explicit_solution);
    }
  else
    {
      initializer(distributed_solution);
      initializer(distributed_solution_dot);
      initializer(distributed_explicit_solution);
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

      explicit_solution = solution;
      distributed_explicit_solution = solution;

    }
}


template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::update_functions_and_constraints (const double &t)
{
  dirichlet_bcs.set_time(t);
  dirichlet_bcs_dot.set_time(t);
  forcing_terms.set_time(t);
  neumann_bcs.set_time(t);

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

  constraints_dot.close ();
}




template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::assemble_matrices (const double t,
                                                     const typename LAC::VectorType &solution,
                                                     const typename LAC::VectorType &solution_dot,
                                                     const double alpha)
{
  auto _timer = computing_timer.scoped_timer ("Assemble matrices");
  update_functions_and_constraints(t);
  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);


  typename LAC::VectorType tmp(solution);
  typename LAC::VectorType tmp_dot(solution_dot);
  constraints.distribute(tmp);
  constraints_dot.distribute(tmp_dot);

  distributed_solution = tmp;
  distributed_solution_dot = tmp_dot;

  if (old_t < t)
    {
      explicit_solution.reinit(solution);
      explicit_solution = solution;

      distributed_explicit_solution.reinit(distributed_solution);
      distributed_explicit_solution = distributed_solution;

      old_t = t;
    }


  interface.initialize_data(*dof_handler,
                            distributed_solution,
                            distributed_solution_dot,
                            distributed_explicit_solution,
                            t, alpha);

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
                              LATrilinos::VectorType &y_expl,
                              LATrilinos::VectorType &distributed_y,
                              LATrilinos::VectorType &distributed_y_dot,
                              LATrilinos::VectorType &distributed_y_expl,
                              bool adaptive_refinement)
{
  distributed_y = y;
  distributed_y_dot = y_dot;
  distributed_y_expl = y_expl;

  parallel::distributed::SolutionTransfer<dim, LATrilinos::VectorType, DoFHandler<dim,spacedim> > sol_tr(*dof_handler);

  std::vector<const LATrilinos::VectorType *> old_sols (3);
  old_sols[0] = &distributed_y;
  old_sols[1] = &distributed_y_dot;
  old_sols[2] = &distributed_y_expl;

  triangulation->prepare_coarsening_and_refinement();
  sol_tr.prepare_for_coarsening_and_refinement (old_sols);

  if (adaptive_refinement)
    triangulation->execute_coarsening_and_refinement ();
  else
    triangulation->refine_global (1);

  setup_dofs(false);

  LATrilinos::VectorType new_sol (y);
  LATrilinos::VectorType new_sol_dot (y_dot);
  LATrilinos::VectorType new_sol_expl (y_expl);

  std::vector<LATrilinos::VectorType *> new_sols (3);
  new_sols[0] = &new_sol;
  new_sols[1] = &new_sol_dot;
  new_sols[2] = &new_sol_expl;

  sol_tr.interpolate (new_sols);

  y = new_sol;
  y_dot = new_sol_dot;
  y_expl = new_sol_expl;
}

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::
refine_and_transfer_solutions(LADealII::VectorType &y,
                              LADealII::VectorType &y_dot,
                              LADealII::VectorType &y_expl,
                              LADealII::VectorType &,
                              LADealII::VectorType &,
                              LADealII::VectorType &,
                              bool adaptive_refinement)
{
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

  LADealII::VectorType new_sol;
  LADealII::VectorType new_sol_dot;
  LADealII::VectorType new_sol_expl;

  std::vector<LADealII::VectorType> new_sols (3);

  new_sols[0].reinit(y);
  new_sols[1].reinit(y_dot);
  new_sols[2].reinit(y_expl);

  sol_tr.interpolate (old_sols, new_sols);

  y      = new_sols[0];
  y_dot  = new_sols[1];
  y_expl = new_sols[2];

}

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::refine_mesh ()
{
  auto _timer = computing_timer.scoped_timer ("Mesh refinement");

  if (adaptive_refinement)
    {
      Vector<float> estimated_error_per_cell (triangulation->n_active_cells());

      interface.estimate_error_per_cell(*dof_handler,
                                        distributed_solution,
                                        estimated_error_per_cell);

      pgr.mark_cells(estimated_error_per_cell, *triangulation);
    }

  refine_and_transfer_solutions(solution,
                                solution_dot,
                                explicit_solution,
                                distributed_solution,
                                distributed_solution_dot,
                                distributed_explicit_solution,
                                adaptive_refinement);

  distributed_explicit_solution = explicit_solution;

  old_t = -std::numeric_limits<double>::max();
}

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::make_grid_fe()
{
  triangulation = SP(pgg.distributed(comm));
  dof_handler = SP(new DoFHandler<dim, spacedim>(*triangulation));
  interface.postprocess_newly_created_triangulation(*triangulation);
  fe = SP(interface.pfe());
  triangulation->refine_global (initial_global_refinement);
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
        euler.start_ode(solution, solution_dot);

      eh.error_from_exact(interface.get_mapping(), *dof_handler, distributed_solution, exact_solution);
    }

  eh.output_table(pcout);

  if (output_timer)
    {
      pcout << std::endl;
      Teuchos::TimeMonitor::summarize(std::cout,true);
    }
}

/*** ODE Argument Interface ***/

template <int dim, int spacedim, typename LAC>
shared_ptr<typename LAC::VectorType>
piDoMUS<dim, spacedim, LAC>::create_new_vector() const
{
  shared_ptr<typename LAC::VectorType> ret = SP(new typename LAC::VectorType(solution));
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

  update_functions_and_constraints(t);

  typename LAC::VectorType tmp(solution);
  typename LAC::VectorType tmp_dot(solution_dot);
  constraints.distribute(tmp);
  constraints_dot.distribute(tmp_dot);
  distributed_solution = tmp;
  distributed_solution_dot = tmp_dot;


  interface.initialize_data(*dof_handler,
                            distributed_solution,
                            distributed_solution_dot,
                            distributed_explicit_solution,
                            t, 0.0);

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
  if (use_space_adaptivity)
    {
      double max_kelly=0;
      auto _timer = computing_timer.scoped_timer ("Compute error estimator");
      update_functions_and_constraints(t);

      typename LAC::VectorType tmp_c(solution);
      constraints.distribute(tmp_c);
      distributed_solution = tmp_c;

      Vector<float> estimated_error_per_cell (triangulation->n_active_cells());

      interface.estimate_error_per_cell(*dof_handler,
                                        distributed_solution,
                                        estimated_error_per_cell);

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
                                        explicit_solution,
                                        distributed_solution,
                                        distributed_solution_dot,
                                        distributed_explicit_solution,
                                        adaptive_refinement);

          //    old_t = -std::numeric_limits<double>::max();

          MPI::COMM_WORLD.Barrier();

          return true;
        }
      else // if max_kelly > kelly_threshold
        {

          return false;
        }

    }
  else // use space adaptivity

    return false;
}


template <int dim, int spacedim, typename LAC>
int
piDoMUS<dim, spacedim, LAC>::residual(const double t,
                                      const typename LAC::VectorType &solution,
                                      const typename LAC::VectorType &solution_dot,
                                      typename LAC::VectorType &dst)
{
  auto _timer = computing_timer.scoped_timer ("Residual");
  update_functions_and_constraints(t);

  typename LAC::VectorType tmp(solution);
  typename LAC::VectorType tmp_dot(solution_dot);
  constraints.distribute(tmp);
  constraints_dot.distribute(tmp_dot);

  distributed_solution = tmp;
  distributed_solution_dot = tmp_dot;

  if (old_t < t)
    {
      explicit_solution.reinit(solution);
      explicit_solution = solution;

      distributed_explicit_solution.reinit(distributed_solution);
      distributed_explicit_solution = distributed_solution;

      old_t = t;
    }


  interface.initialize_data(*dof_handler,
                            distributed_solution,
                            distributed_solution_dot,
                            distributed_explicit_solution,
                            t, 0.0);

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
        dst[j] = solution(j) - distributed_solution(j);
    }

  dst.compress(VectorOperation::insert);

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
                 typename SolverFGMRES<typename LAC::VectorType>::AdditionalData(50, true));

          auto S_inv = inverse_operator(jacobian_op, solver, jacobian_preconditioner_op);
          S_inv.vmult(dst, src);

          tot_iteration += solver_control.last_step();
        }
      catch (...)
        {
          unsigned int solver_iterations = matrices[0]->m();
          if ( max_iterations_finer != 0 )
            solver_iterations = max_iterations_finer;

          if (enable_finer_preconditioner)
            {
              if (verbose)
                pcout << " --> Finer preconditioner "
                      << std::endl;
              PrimitiveVectorMemory<typename LAC::VectorType> mem;
              SolverControl solver_control (solver_iterations,
                                            jacobian_solver_tolerance);

              SolverFGMRES<typename LAC::VectorType>
              solver(solver_control, mem,
                     typename SolverFGMRES<typename LAC::VectorType>::AdditionalData(80, true));

              auto S_inv = inverse_operator(jacobian_op, solver, jacobian_preconditioner_op_finer);
              S_inv.vmult(dst, src);

              tot_iteration += solver_control.last_step();
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

  assemble_matrices(t, src_yy, src_yp, alpha);
  if (use_direct_solver == false)
    {
      auto _timer = computing_timer.scoped_timer ("Compute system operators");

      interface.compute_system_operators(matrices,
                                         jacobian_op,
                                         jacobian_preconditioner_op,
                                         jacobian_preconditioner_op_finer);
    }

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

  set_constrained_dofs_to_zero(diff_comps);
  return diff_comps;
}



template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::set_constrained_dofs_to_zero(typename LAC::VectorType &v) const
{
  for (unsigned int i = 0; i < global_partitioning.n_elements(); ++i)
    {
      auto j = global_partitioning.nth_index_in_set(i);
      if (constraints.is_constrained(j))
        v[j] = 0;
    }
}

template class piDoMUS<2, 2, LATrilinos>;
template class piDoMUS<2, 3, LATrilinos>;
template class piDoMUS<3, 3, LATrilinos>;

template class piDoMUS<2, 2, LADealII>;
template class piDoMUS<2, 3, LADealII>;
template class piDoMUS<3, 3, LADealII>;

