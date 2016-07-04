#ifndef __pidomus_lambdas_h_
#define __pidomus_lambdas_h_

#include "lac/lac_type.h"
#include <deal2lkit/utilities.h>

using namespace deal2lkit;
// forward declaration
template <int dim, int spacedim, typename LAC> class piDoMUS;

/**
 * Lambdas class. A piDoMUS object has a Lambdas object and the
 * std::functions of the provided stepper (e.g., ida, imex)
 * are set to be equal to those here implemented.
 *
 * By default, the std::functions of this class call the
 * namesake functions of the piDoMUS object, which is specified
 * either when the Lambdas object is constructed or by calling the
 * Lambdas::initialize_simulator() function.
 *
 * The aim of this class is to increase the flexibility of piDoMUS.
 * piDoMUS offers flexibility by itself thanks to the signals
 * to whom the user can connect in order to perform problem-specific
 * tasks. Whether the behavior of a function should be completely different,
 * the user can ovveride the functions declared here (without the need
 * of modifying piDoMUS' source code).
 */

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
   * Destructor. Does nothing.
   */
  ~Lambdas ();

  /**
   * Initialize this class for a given simulator.
   *
   * @param simulator_object A reference to the main simulator object.
   */
  void initialize_simulator (piDoMUS<dim, spacedim, LAC> &simulator_object);

  /**
   * Set the default behavior of the functions of this class,
   * which is call the namesake functions of the simulator_bject.
   */
  void set_functions_to_default();

  /**
   * Return a shared_ptr<VECTOR_TYPE>. A shared_ptr is needed in order to
   * keep the pointed vector alive, without the need to use a static variable.
   */
  std::function<shared_ptr<typename LAC::VectorType>()> create_new_vector;

  /** Compute residual. */
  std::function<int(const double t,
                    const typename LAC::VectorType &y,
                    const typename LAC::VectorType &y_dot,
                    typename LAC::VectorType &res)> residual;

  /** Compute Jacobian. */
  std::function<int(const double t,
                    const typename LAC::VectorType &y,
                    const typename LAC::VectorType &y_dot,
                    const double alpha)> setup_jacobian;

  /** Solve linear system. */
  std::function<int(const typename LAC::VectorType &rhs, typename LAC::VectorType &dst)> solve_jacobian_system;

  /**
   * Store solutions to file.
   */
  std::function<void (const double t,
                      const typename LAC::VectorType &sol,
                      const typename LAC::VectorType &sol_dot,
                      const unsigned int step_number)> output_step;

  /**
   * Evaluate if the mesh should be refined or not. If so,
   * it refines and interpolate the solutions from the old to the
   * new mesh.
   */
  std::function<bool (const double t,
                      typename LAC::VectorType &sol,
                      typename LAC::VectorType &sol_dot)> solver_should_restart;

  /**
   * Return a vector whose component are 1 if the corresponding
   * dof is differential, 0 if algebraic. This function is needed
   * by the IDAInterface stepper.
   */
  std::function<typename LAC::VectorType&()> differential_components;

  /**
   * Return a vector whose components are the weights used by
   * IDA to compute the norm. By default this function is not
   * implemented.
   */
  std::function<typename LAC::VectorType&()> get_local_tolerances;

  /**
   * Return a vector which is a lumped mass matrix. This function
   * is used by Kinsol (through imex) for setting the weights used
   * for computing the norm a vector.
   */
  std::function<typename LAC::VectorType&()> get_lumped_mass_matrix;

  /**
   * Compute the matrix-vector product Jacobian times @p src,
   * and the result is put in @p dst.
   */
  std::function<int(const typename LAC::VectorType &src,
                    typename LAC::VectorType &dst)> jacobian_vmult;

  /**
   * A pointer to the simulator object to which we want to get
   * access.
   */
  piDoMUS<dim,spacedim,LAC> *simulator;
};

#endif
