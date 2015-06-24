#ifndef dae_time_integrator_h
#define dae_time_integrator_h

#include <deal.II/base/logstream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

// For time integration.
#include <ida/ida.h>
#include <ida/ida_spils.h>
#include <ida/ida_spgmr.h>
#include <ida/ida_spbcgs.h>
#include <ida/ida_sptfqmr.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>


#include "ode_argument.h"
#include "parameter_acceptor.h"

template<typename VEC=Vector<double> >
class DAETimeIntegrator : public ParameterAcceptor
{
public:
  /** Constructor for the DAETimeIntegrator class. The Solver class is
   * required to have a Solver.solve(VEC &dst, const
   * VEC &src) method that will be called by the time
   * integrator to find out about the solution to a given src. */
  DAETimeIntegrator(OdeArgument<VEC> &solver);

  /** House cleaning. */
  ~DAETimeIntegrator();

  /** Declare parameters for this class to function properly. */
  virtual void declare_parameters(ParameterHandler &prm);

  /** Evolve. This function returns the final number of steps. */
  unsigned int start_ode(VEC &solution,
                         VEC &solution_dot,
                         const unsigned int max_steps);

  /** Clear internal memory, and
  start with clean
  objects. This is useful if
  you need to refine your
  mesh between stesp. */
  void reset_ode(const double t, VEC &y, VEC &yp,
                 double h, unsigned int max_steps);


  /** Final time. */
  double final_time;

  /** Initial time for the ode.*/
  double initial_time;

private:
  /** The bubble membrane poperties. */
  OdeArgument<VEC> &solver;

  /** Initial step size. */
  double initial_step_size;

  /** Minimum step size. */
  double min_step_size;

  /** Absolute error tolerance for adaptive time stepping. */
  double abs_tol;

  /** Relative error tolerance for adaptive time stepping. */
  double rel_tol;

  /** Maximum number of time steps. */
  unsigned int max_n_steps;

  double outputs_period;

  /** Type of initial conditions. */
  std::string ic_type;

  /** Alpha to use in Newton method for IC calculation. */
  double ic_alpha;

  /** Maximum number of iterations for Newton method in IC calculation. */
  unsigned ic_max_iter;

  /** Initialization flag.*/
  bool is_initialized;

  /** If true, we use
  preconditioned gmres. */
  std::string iterative_solver_type;

  /** Provides the Jacobian vector
  product. If this is false,
  then finite difference is
  used. */
  bool provide_jac;
  /** Provide preconditioning for
  Jacobian. If not set, then no
  preconditioning is used. */
  bool provide_jac_prec;

  /** Use local tolerances when computing absolute tolerance. */
  bool use_local_tolerances;

  /** Ida memory object. */
  void *ida_mem;

  /** Ida solution vector. */
  N_Vector yy;
  /** Ida solution derivative vector. */
  N_Vector yp;
  /** Ida absolute tolerances vector. */
  N_Vector abs_tolls;
  /** Ida differential components vector. */
  N_Vector diff_id;


};

template <>
unsigned int DAETimeIntegrator<Vector<double> >::start_ode(Vector<double> &solution,
                                                           Vector<double> &solution_dot,
                                                           const unsigned int max_steps);

template <>
void DAETimeIntegrator<Vector<double> >::reset_ode(double current_time,
                                                   Vector<double> &solution,
                                                   Vector<double> &solution_dot,                                                   double current_time_step,
                                                   unsigned int max_steps);

#endif
