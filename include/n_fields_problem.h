#ifndef _N_FIELDS_LINEAR_PROBLEM_
#define _N_FIELDS_LINEAR_PROBLEM_


#include <deal.II/base/timer.h>
// #include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/linear_operator.h>

// #include <deal.II/lac/precondition.h>

#include "assembly.h"
#include "interface.h"
#include "parsed_grid_generator.h"
#include "parsed_finite_element.h"
#include "error_handler.h"
#include "parsed_function.h"
#include "parsed_data_out.h"
#include "parameter_acceptor.h"

#include "sak_data.h"
#include "stokes_derived_interface.h"

using namespace dealii;

template <int dim, int spacedim=dim, int n_components=1>
class NFieldsProblem : public ParameterAcceptor
{

  // This is a class required to make tests
  template<int fdim, int fspacedim>
  friend void test(NFieldsProblem<fdim,fspacedim> &);

public:

  enum RefinementMode
  {
    global_refinement=0,
    adaptive_refinement=1
  };

  NFieldsProblem (const Interface<dim,spacedim,n_components> &energy);

  virtual void declare_parameters(ParameterHandler &prm);

  void run ();

private:
  void make_grid_fe();
  void setup_dofs (const bool initial_step);
  void setup_dofs ();
  void assemble_preconditioner ();
  void build_preconditioner ();
  void assemble_system ();
  void solve ();
  void output_results ();
  //void refine_mesh (const unsigned int max_grid_level);
  void refine_mesh ();
  double compute_residual(const double alpha); // const;
  double determine_step_length () const;
  void process_solution ();

  unsigned int n_cycles;
  unsigned int initial_global_refinement;

  const Interface<dim,spacedim,n_components>    &energy;
  ConditionalOStream        pcout;
  std::ofstream         timer_outfile;
  ConditionalOStream        tcout;

  shared_ptr<parallel::distributed::Triangulation<dim,spacedim> > triangulation;
  const MappingQ<dim,spacedim>                   mapping;

  shared_ptr<FiniteElement<dim,spacedim> >       fe;
  shared_ptr<DoFHandler<dim,spacedim> >          dof_handler;

  ConstraintMatrix                          constraints;

  TrilinosWrappers::BlockSparseMatrix       matrix;
  TrilinosWrappers::BlockSparseMatrix       preconditioner_matrix;

  LinearOperator<TrilinosWrappers::MPI::BlockVector> preconditioner_op;
  LinearOperator<TrilinosWrappers::MPI::BlockVector> system_op;

  TrilinosWrappers::MPI::BlockVector        solution;
  TrilinosWrappers::MPI::BlockVector        old_solution;
  TrilinosWrappers::MPI::BlockVector        present_solution;
  TrilinosWrappers::MPI::BlockVector        newton_update;
  TrilinosWrappers::MPI::BlockVector        rhs;

  bool                                      rebuild_matrix;
  bool                                      rebuild_preconditioner;

  TimerOutput                               computing_timer;

  void setup_matrix ( const std::vector<IndexSet> &partitioning,
                      const std::vector<IndexSet> &relevant_partitioning);
  void setup_preconditioner ( const std::vector<IndexSet> &partitioning,
                              const std::vector<IndexSet> &relevant_partitioning);


  void
  local_assemble_preconditioner (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                 Assembly::Scratch::NFields<dim,spacedim> &scratch,
                                 Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> &data);

  void
  copy_local_to_global_preconditioner (const Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> &data);


  void
  local_assemble_system (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                         Assembly::Scratch::NFields<dim,spacedim>  &scratch,
                         Assembly::CopyData::NFieldsSystem<dim,spacedim> &data);

  void
  copy_local_to_global_system (const Assembly::CopyData::NFieldsSystem<dim,spacedim> &data);

  ErrorHandler<1>       eh;
  ParsedGridGenerator<dim,spacedim>   pgg;

  ParsedFunction<spacedim, n_components>        boundary_conditions;
  ParsedFunction<spacedim, n_components>        right_hand_side;
  ParsedFunction<spacedim, n_components>        exact_solution;

  ParsedDataOut<dim, spacedim>                  data_out;
};

#endif
