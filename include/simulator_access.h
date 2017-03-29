#ifndef _pidomus_simulator_acess_h
#define _pidomus_simulator_acess_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>

using namespace dealii;
//using namespace deal2lkit;


// forward declaration

template <int dim, int spacedim, typename LAC> class piDoMUS;
template <int dim, int spacedim, typename LAC> struct Signals;

template <int dim, int spacedim, typename LAC>
class SimulatorAccess
{
public:

  /**
   * Default constructor. Initialize the SimulatorAccess object without
   * a reference to a particular piDoMUS object. You will later have
   * to call initialize() to provide this reference to the piDoMUS
   * object.
   */
  SimulatorAccess ();

  /**
   * Create a SimulatorAccess object that is already initialized for
   * a particular piDoMUS.
   */
  SimulatorAccess (const piDoMUS<dim,spacedim,LAC> &simulator_object);

  /**
   * Destructor. Does nothing but is virtual so that derived classes
   * destructors are also virtual.
   */
  virtual
  ~SimulatorAccess ();

  /**
   * Initialize this class for a given simulator. This function is marked
   * as virtual so that derived classes can do something upon
   * initialization as well, for example look up and cache data; derived
   * classes should call this function from the base class as well,
   * however.
   *
   * @param simulator_object A reference to the main simulator object.
   */
  virtual void initialize_simulator (const piDoMUS<dim,spacedim,LAC> &simulator_object) const;

  /** @name Accessing variables that identify overall properties of the simulator */
  /** @{ */

  /**
   * Return a reference to the piDoMUS itself. Note that you can not
   * access any members or functions of the piDoMUS. This function
   * exists so that any class with SimulatorAccess can create other
   * objects with SimulatorAccess (because initializing them requires a
   * reference to the piDoMUS).
   */
  const piDoMUS<dim,spacedim,LAC> &
  get_simulator () const;

  /**
   * Get access to the structure containing the signals of piDoMUS
   */
  Signals<dim,spacedim,LAC> &
  get_signals() const;

#ifdef DEAL_II_WITH_MPI
  /**
   * Return the MPI communicator for this simulation.
   */
  MPI_Comm
  get_mpi_communicator () const;
#endif

  /**
   * Return a reference to the stream object that only outputs something
   * on one processor in a parallel program and simply ignores output put
   * into it on all other processors.
   */
  const ConditionalOStream &
  get_pcout () const;

  /**
   * Return the current simulation time in seconds.
   */
  double get_current_time () const;

  /**
   * Return the current alpha in the expression
   * \f$\dot{y}=\alpha y + \beta \bar{y}\f$
   */
  double get_alpha () const;

  /**
   * Return the size of the current time step.
   */
  double
  get_timestep () const;

  /**
   * Return the current number of a time step.
   */
  unsigned int
  get_timestep_number () const;

  /**
   * Return a reference to the triangulation in use by the simulator
   * object.
   */
  const parallel::distributed::Triangulation<dim,spacedim> &
  get_triangulation () const;
  /** @} */


  /** @name Accessing variables that identify the solution of the problem */
  /** @{ */


  /**
   * Return a reference to the vector that has the current
   * solution of the entire system. This vector is associated with
   * the DoFHandler object returned by get_dof_handler().
   */
  const typename LAC::VectorType &
  get_solution () const;

  /**
   * Return a reference to the vector that has the solution_dot of
   * the entire system. This vector is associated with the
   * DoFHandler object returned by get_dof_handler().
   */
  const typename LAC::VectorType &
  get_solution_dot () const;

  /**
   * Return a reference to the vector that has the current
   * solution of the entire system. This vector is associated with
   * the DoFHandler object returned by get_dof_handler().
   *
   * @note In general the vector is a distributed vector; however, it
   * contains ghost elements for all locally relevant degrees of freedom.
   */
  const typename LAC::VectorType &
  get_locally_relevant_solution () const;

  /**
   * Return a reference to the vector that has the solution_dot of
   * the entire system. This vector is associated with the
   * DoFHandler object returned by get_dof_handler().
   *
   * @note In general the vector is a distributed vector; however, it
   * contains ghost elements for all locally relevant degrees of freedom.
   */
  const typename LAC::VectorType &
  get_locally_relevant_solution_dot () const;

  /**
   * Return a reference to the vector that has the solution of the
   * entire system at the previous time step. This vector is
   * associated with the DoFHandler object returned by
   * get_dof_handler().
   *
   * @note In general the vector is a distributed vector; however, it
   * contains ghost elements for all locally relevant degrees of freedom.
   */
  const typename LAC::VectorType &
  get_locally_relevant_explicit_solution () const;

  /**
   * Return a reference to the DoFHandler that is used to
   * discretize the variables at the current time step.
   */
  const DoFHandler<dim,spacedim> &
  get_dof_handler () const;

  /**
   * Return a reference to the finite element that the DoFHandler
   * that is used to discretize the variables at the current time
   * step is built on.
   */
  const FiniteElement<dim,spacedim> &
  get_fe () const;

  /**
   * Return a reference to the ParsedDirichletBCs that stores
   * the Dirichlet boundary conditions set in the parameter file.
   */
  const ParsedDirichletBCs<dim,spacedim> &
  get_dirichlet_bcs () const;

  /**
   * Return a reference to the boolean, which defines if local
   * refinement is performed or not.
   */
  const bool &
  get_transient_refinement () const;

  /**
   * Return a reference to the boolean, which defines if local
   * refinement is performed or not.
   */
  const shared_ptr<SolverControl> &
  get_solver_control () const;

  /** @} */

private:

  /**
   * A pointer to the simulator object to which we want to get
   * access.
   */
  mutable const piDoMUS<dim,spacedim,LAC> *simulator;
};



#endif
