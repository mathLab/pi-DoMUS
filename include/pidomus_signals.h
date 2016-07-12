#ifndef _pidomus_signals_h
#define _pidomus_signals_h

#include "simulator_access.h"

/**
 * A class that collects the definition of signals that can be triggered
 * at different points in a computation. A signal is in essence an event
 * that is triggered whenever the program passes a certain point in a
 * computation. Parties interested in any of these signals can attach
 * "slots" to a signal. A slot is, in essence, a function that is called
 * whenever the signal is triggered. Multiple slots (or none) can be
 * attached to the same signal. To be as general as possible, slots are
 * not actually just pointers to functions, but std::function objects
 * that have a certain signature. Consequently, they can have much more
 * complicated types than just function pointers, such as objects with
 * an <code>operator()</code> or function calls treated with things
 * like std::bind.
 *
 * The documentation of each of the signals below indicates when
 * exactly it is called.
 *
 */
template <int dim, int spacedim, typename LAC>
struct Signals
{

  /**
  * This signal is called after that the initial conditions have
  * been set according to the parameter file.
  *
  * The functions (slots) that can attach to this signal need to
  * take two vectors.
  */
  boost::signals2::signal<void (typename LAC::VectorType &y,
                                typename LAC::VectorType &y_dot)> fix_initial_conditions;

  /**
   * This signal is called when the constraint matrices are buit,
   * i.e., when the setup_dofs() is called and when a time step is
   * done and so the constraints, which might be time dependent,
   * must be updated. Specifically, it is called inside the
   * function update_functions_and_constraints() after that the
   * hanging nodes have been computed and after that the boundary values
   * have been included. Note that ConstraintMatrix::close() will be called
   * after this signal, so no need to call it inside.
   *
   * The functions that can attach to this signal must take two
   * ConstraintMatrix.
   */
  boost::signals2::signal<void (std::vector<shared_ptr<ConstraintMatrix> > &constraints,
                                ConstraintMatrix &constraints_dot)> update_constraint_matrices;

  /**
   * This signal is called in the make_grid_and_fe()
   * function. Afther this, the global refinement given in the
   * parameter file is performed.
   *
   * The functions that can attach to this signal must take a
   * Triangulation as argument.
   */
  boost::signals2::signal<void (typename parallel::distributed::Triangulation<dim,spacedim> &)>
  postprocess_newly_created_triangulation;


  /**
   * This signal is called after that a mesh refinement is performed
   * and the solutions have been interpolated to the new mesh.
   *
   * The functions that can attach to this signal must take two vectors.
   */
  boost::signals2::signal<void (typename LAC::VectorType &y,
                                typename LAC::VectorType &y_dot)> fix_solutions_after_refinement;

  /**
   * This signal is called inside the function
   * differential_components(). It allows to fix the algebraic
   * components that cannot be specified through the parameter file.
   *
   * The functions that can attach to this signal must take one vector as argument.
   */
  boost::signals2::signal<void (typename LAC::VectorType &diff_comp)> fix_differential_components;

  // the following signals are called at begin and end of the
  // corresponding function in pi-DoMUS. They should be used for debug
  // purposes

  boost::signals2::signal<void ()> begin_make_grid_fe;
  boost::signals2::signal<void ()> begin_setup_dofs;
  boost::signals2::signal<void ()> begin_refine_mesh;
  boost::signals2::signal<void ()> begin_setup_jacobian;
  boost::signals2::signal<void ()> begin_residual;
  boost::signals2::signal<void ()> begin_solve_jacobian_system;
  boost::signals2::signal<void ()> begin_refine_and_transfer_solutions;
  boost::signals2::signal<void ()> begin_assemble_matrices;
  boost::signals2::signal<void ()> begin_solver_should_restart;

  boost::signals2::signal<void ()> end_make_grid_fe;
  boost::signals2::signal<void ()> end_setup_dofs;
  boost::signals2::signal<void ()> end_refine_mesh;
  boost::signals2::signal<void ()> end_setup_jacobian;
  boost::signals2::signal<void ()> end_residual;
  boost::signals2::signal<void ()> end_solve_jacobian_system;
  boost::signals2::signal<void ()> end_refine_and_transfer_solutions;
  boost::signals2::signal<void ()> end_assemble_matrices;
  boost::signals2::signal<void ()> end_solver_should_restart;

};

#endif
