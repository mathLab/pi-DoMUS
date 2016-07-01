#ifndef __pidomus_lambdas_h_
#define __pidomus_lambdas_h_

#include "lac/lac_type.h"
#include <deal2lkit/utilities.h>

using namespace deal2lkit;
// forward declaration
template <int dim, int spacedim, typename LAC> class piDoMUS;

template <int dim, int spacedim, typename LAC>
class Lambdas
{
public:

  /**
   * Default constructor. Initialize the Lambdas object without
   * a reference to a particular piDoMUS object. You will later have
   * to call initialize() to provide this reference to the piDoMUS
   * object.
   */
  Lambdas ();

  /**
   * Create a Lambdas object that is already initialized for
   * a particular piDoMUS.
   */
  Lambdas (piDoMUS<dim,spacedim,LAC> &simulator_object);

  /**
   * Destructor. Does nothing but is virtual so that derived classes
   * destructors are also virtual.
   */
  virtual
  ~Lambdas ();

  /**
   * Initialize this class for a given simulator. This function is marked
   * as virtual so that derived classes can do something upon
   * initialization as well, for example look up and cache data; derived
   * classes should call this function from the base class as well,
   * however.
   *
   * @param simulator_object A reference to the main simulator object.
   */
  virtual void initialize_simulator (piDoMUS<dim, spacedim, LAC> &simulator_object);

  /**
   * Set the default behavior of the functions of this class,
   * which is call the namesake functions of the simulator_bject.
   */
  void set_functions_to_default();

  std::function<shared_ptr<typename LAC::VectorType>()> create_new_vector;

  /** standard function computing residuals */
  std::function<int(const double t,
                    const typename LAC::VectorType &y,
                    const typename LAC::VectorType &y_dot,
                    typename LAC::VectorType &res)> residual;

  /** standard function computing the Jacobian */
  std::function<int(const double t,
                    const typename LAC::VectorType &y,
                    const typename LAC::VectorType &y_dot,
                    const double alpha)> setup_jacobian;

  /** standard function solving linear system */
  std::function<int(const typename LAC::VectorType &rhs, typename LAC::VectorType &dst)> solve_jacobian_system;

  std::function<void (const double t,
                      const typename LAC::VectorType &sol,
                      const typename LAC::VectorType &sol_dot,
                      const unsigned int step_number)> output_step;

  std::function<bool (const double t,
                      typename LAC::VectorType &sol,
                      typename LAC::VectorType &sol_dot)> solver_should_restart;

  std::function<typename LAC::VectorType&()> differential_components;

  std::function<typename LAC::VectorType&()> get_local_tolerances;

  std::function<typename LAC::VectorType&()> get_lumped_mass_matrix;

  /**
   * A pointer to the simulator object to which we want to get
   * access.
   */
  piDoMUS<dim,spacedim,LAC> *simulator;
};

#endif
