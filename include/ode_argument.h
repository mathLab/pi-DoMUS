#ifndef __thiwi__ode_argument_h
#define __thiwi__ode_argument_h

#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_view.h>

#include "utilities.h"

using namespace dealii;

/** Base class that needs to be inherited by any function that wants
 * to use the time integrator class. */
template<typename VEC=Vector<double> >
class OdeArgument
{

public :

  OdeArgument(const MPI_Comm &communicator) :
    communicator(communicator) {};

  const MPI_Comm &get_comm() const
  {
    return communicator;
  }


  virtual shared_ptr<VEC>
  create_new_vector() const = 0;

  /** Returns the number of degrees of freedom. Pure virtual function. */
  virtual unsigned int n_dofs() const = 0;

  /** This function is called at the end of each iteration step for
   * the ode solver. Once again, the conversion between pointers and
   * other forms of vectors need to be done inside the inheriting
   * class. */
  virtual void output_step(VEC &solution,
                           VEC &solution_dot,
                           const double t,
                           const unsigned int step_number,
                           const double h) = 0;

  /** This function will check the behaviour of the solution. If it
   * is converged or if it is becoming unstable the time integrator
   * will be stopped. If the convergence is not achived the
   * calculation will be continued. If necessary, it can also reset
   * the time stepper. */
  virtual bool solution_check(VEC &solution,
                              VEC &solution_dot,
                              const double t,
                              const unsigned int step_number,
                              const double h) = 0;

  /** For dae problems, we need a
   residual function. */
  virtual int residual(const double t,
                       VEC &dst,
                       const VEC &src_yy,
                       const VEC &src_yp) = 0;

  /** Jacobian vector product. */
  virtual int jacobian(const double t,
                       VEC &dst,
                       const VEC &src_yy,
                       const VEC &src_yp,
                       const VEC &src,
                       const double alpha);

  /** Setup Jacobian preconditioner. */
  virtual int setup_jacobian_prec(const double t,
                                  const VEC &src_yy,
                                  const VEC &src_yp,
                                  const double alpha);

  /** Jacobian preconditioner
   vector product. */
  virtual int jacobian_prec(const double t,
                            VEC &dst,
                            const VEC &src_yy,
                            const VEC &src_yp,
                            const VEC &src,
                            const double alpha);

  /** And an identification of the
   differential components. This
   has to be 1 if the
   corresponding variable is a
   differential component, zero
   otherwise.  */
  virtual VEC &differential_components();

  virtual VEC &get_local_tolerances();

  virtual ~OdeArgument() {};

  bool reset_time_integrator;

  bool stop_time_integrator;

  const MPI_Comm &communicator;

};

#endif
