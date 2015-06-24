#include "../include/dae_time_integrator.h"
#include "../include/ode_argument.h"

#include <deal.II/base/utilities.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <iostream>

#include <nvector/nvector_parallel.h>
#include <nvector/nvector_serial.h>

using namespace dealii;
using namespace std;


void copy(TrilinosWrappers::MPI::BlockVector &dst, const N_Vector &src)
{
  IndexSet is = dst.locally_owned_elements();
  AssertDimension(is.n_elements(), NV_LOCLENGTH_P(src));
  for (unsigned int i=0; i<is.n_elements(); ++i)
    {
      dst[is.nth_index_in_set(i)] = NV_Ith_P(src, i);
    }
}

void copy(N_Vector &dst, const TrilinosWrappers::MPI::BlockVector &src)
{
  IndexSet is = src.locally_owned_elements();
  AssertDimension(is.n_elements(), NV_LOCLENGTH_P(dst));
  for (unsigned int i=0; i<is.n_elements(); ++i)
    {
      NV_Ith_P(dst, i) = src[is.nth_index_in_set(i)];
    }
}

template<typename VEC>
int t_dae_residual(realtype tt, N_Vector yy, N_Vector yp,
                   N_Vector rr, void *user_data)
{
  OdeArgument<VEC> &solver = *static_cast<OdeArgument<VEC> *>(user_data);

  shared_ptr<VEC> src_yy = solver.create_new_vector();
  shared_ptr<VEC> src_yp = solver.create_new_vector();
  shared_ptr<VEC> residual = solver.create_new_vector();

  copy(*src_yy, yy);
  copy(*src_yp, yp);

  int err = solver.residual(tt, *src_yy, *src_yp, *residual);

  copy(rr, *residual);

  return err;
}

int dae_residual(realtype tt, N_Vector yy, N_Vector yp,
                 N_Vector rr, void *user_data)
{
  OdeArgument<Vector<double> > &solver = *static_cast<OdeArgument<Vector<double> > *>(user_data);
  Assert(NV_LENGTH_S(yy) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(yy), solver.n_dofs()));
  Assert(NV_LENGTH_S(yp) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(yp), solver.n_dofs()));
  //    if ((NV_LENGTH_S(yy) == NV_LENGTH_S(yp)) & (NV_LENGTH_S(yy) =! NV_LENGTH_S(rr)))
  NV_LENGTH_S(rr) = NV_LENGTH_S(yy);

  Assert(NV_LENGTH_S(rr) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(rr), solver.n_dofs()));

  const VectorView<double> src_yy(solver.n_dofs(), NV_DATA_S(yy));
  const VectorView<double> src_yp(solver.n_dofs(), NV_DATA_S(yp));
  VectorView<double> residual(solver.n_dofs(), NV_DATA_S(rr));
  return solver.residual(tt, src_yy, src_yp, residual);
}


template<typename VEC>
int t_dae_setup_prec(realtype tt, // time
                     N_Vector yy,
                     N_Vector yp,
                     N_Vector rr, // Current residual
                     realtype alpha, // J = dG/dyy + alpha dG/dyp
                     void *user_data, // the pointer to the correct class
                     N_Vector /*tmp1*/, // temporary storage
                     N_Vector /*tmp2*/,
                     N_Vector /*tmp3*/)
{
  OdeArgument<VEC > &solver = *static_cast<OdeArgument<VEC > *>(user_data);
  // A previous call to residual has already been done.

  shared_ptr<VEC> src_yy = solver.create_new_vector();
  shared_ptr<VEC> src_yp = solver.create_new_vector();

  copy(*src_yy, yy);
  copy(*src_yp, yp);

  return solver.setup_jacobian_prec(tt, *src_yy, *src_yp, alpha);
}



int dae_setup_prec(realtype tt, // time
                   N_Vector yy,
                   N_Vector yp,
                   N_Vector rr, // Current residual
                   realtype alpha, // J = dG/dyy + alpha dG/dyp
                   void *user_data, // the pointer to the correct class
                   N_Vector /*tmp1*/, // temporary storage
                   N_Vector /*tmp2*/,
                   N_Vector /*tmp3*/)
{
  OdeArgument<Vector<double> > &solver = *static_cast<OdeArgument<Vector<double> > *>(user_data);
  Assert(NV_LENGTH_S(yy) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(yy), solver.n_dofs()));
  Assert(NV_LENGTH_S(yp) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(yp), solver.n_dofs()));
  Assert(NV_LENGTH_S(rr) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(rr), solver.n_dofs()));
  // A previous call to residual has already been done.
  const VectorView<double> src_yy(solver.n_dofs(), NV_DATA_S(yy));
  const VectorView<double> src_yp(solver.n_dofs(), NV_DATA_S(yp));
  const VectorView<double> residual(solver.n_dofs(), NV_DATA_S(rr));
  return solver.setup_jacobian_prec(tt, src_yy, src_yp, alpha);
}

template<typename VEC>
int t_dae_jtimes(realtype tt, N_Vector yy, N_Vector yp,
                 N_Vector rr, // Current residual
                 N_Vector src, // right hand side to solve for
                 N_Vector dst, // computed output
                 realtype alpha, // J = dG/dyy + alpha dG/dyp
                 void *user_data, // the pointer to the correct class
                 N_Vector /*tmp*/,
                 N_Vector /*tmp2*/) // Storage
{
  OdeArgument<VEC> &solver = *static_cast<OdeArgument<VEC>*>(user_data);

  shared_ptr<VEC> src_yy = solver.create_new_vector();
  shared_ptr<VEC> src_yp = solver.create_new_vector();
  shared_ptr<VEC> src_v = solver.create_new_vector();
  shared_ptr<VEC> dst_v = solver.create_new_vector();

  copy(*src_yy, yy);
  copy(*src_yp, yp);
  copy(*src_v, src);

  int err = solver.jacobian(tt, *src_yy, *src_yp, alpha, *src_v, *dst_v);
  copy(dst, *dst_v);
  return err;
}


int dae_jtimes(realtype tt, N_Vector yy, N_Vector yp,
               N_Vector rr, // Current residual
               N_Vector src, // right hand side to solve for
               N_Vector dst, // computed output
               realtype alpha, // J = dG/dyy + alpha dG/dyp
               void *user_data, // the pointer to the correct class
               N_Vector /*tmp*/,
               N_Vector /*tmp2*/) // Storage
{
  //double* a = NV_DATA_S(src);
  //for (unsigned int i=0; i<NV_LENGTH_S(src); ++i)
  //    if (!(numbers::is_finite(a[i])))
  //       cout<<i<<endl;
  OdeArgument<Vector<double> > &solver = *static_cast<OdeArgument<Vector<double> >*>(user_data);
  Assert(NV_LENGTH_S(yy) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(yy), solver.n_dofs()));
  Assert(NV_LENGTH_S(yp) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(yp), solver.n_dofs()));
  Assert(NV_LENGTH_S(src) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(src), solver.n_dofs()));
  Assert(NV_LENGTH_S(dst) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(dst), solver.n_dofs()));

  // A previous call to residual has already been done.
  const VectorView<double> src_yy(solver.n_dofs(), NV_DATA_S(yy));
  const VectorView<double> src_yp(solver.n_dofs(), NV_DATA_S(yp));
  const VectorView<double> residual(solver.n_dofs(), NV_DATA_S(rr));
  const VectorView<double> src_v(solver.n_dofs(), NV_DATA_S(src));
  VectorView<double> dst_v(solver.n_dofs(), NV_DATA_S(dst));
  int err = solver.jacobian(tt, src_yy, src_yp, alpha, src_v, dst_v);
  return err;
}


template<typename VEC>
int t_dae_prec(realtype tt, N_Vector yy, N_Vector yp,
               N_Vector rr, // Current residual
               N_Vector src, // right hand side to solve for
               N_Vector dst, // computed output
               realtype alpha, // J = dG/dyy + alpha dG/dyp
               realtype /*delta*/, // input tolerance. The residual rr - Pz has to be smaller than delta
               void *user_data, // the pointer to the correct class
               N_Vector /*tmp*/) // Storage
{
  OdeArgument<VEC > &solver = *static_cast<OdeArgument<VEC>*>(user_data);
  // A previous call to residual has already been done.

  shared_ptr<VEC> src_yy = solver.create_new_vector();
  shared_ptr<VEC> src_yp = solver.create_new_vector();
  shared_ptr<VEC> src_v = solver.create_new_vector();
  shared_ptr<VEC> dst_v = solver.create_new_vector();

  copy(*src_yy, yy);
  copy(*src_yp, yp);
  copy(*src_v, src);

  int err = solver.jacobian_prec(tt, *src_yy, *src_yp, alpha, *src_v, *dst_v);
  copy(dst, *dst_v);
  return err;
}



int dae_prec(realtype tt, N_Vector yy, N_Vector yp,
             N_Vector rr, // Current residual
             N_Vector rvec, // right hand side to solve for
             N_Vector zvec, // computed output
             realtype alpha, // J = dG/dyy + alpha dG/dyp
             realtype /*delta*/, // input tolerance. The residual rr - Pz has to be smaller than delta
             void *user_data, // the pointer to the correct class
             N_Vector /*tmp*/) // Storage
{
  OdeArgument<Vector<double> > &solver = *static_cast<OdeArgument<Vector<double> >*>(user_data);
  Assert(NV_LENGTH_S(yy) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(yy), solver.n_dofs()));
  Assert(NV_LENGTH_S(yp) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(yp), solver.n_dofs()));
  Assert(NV_LENGTH_S(rr) == solver.n_dofs(),
         ExcDimensionMismatch(NV_LENGTH_S(rr), solver.n_dofs()));
  // A previous call to residual has already been done.
  const VectorView<double> src_yy(solver.n_dofs(), NV_DATA_S(yy));
  const VectorView<double> src_yp(solver.n_dofs(), NV_DATA_S(yp));
  const VectorView<double> residual(solver.n_dofs(), NV_DATA_S(rr));
  const VectorView<double> rhs(solver.n_dofs(), NV_DATA_S(rvec));
  VectorView<double> output(solver.n_dofs(), NV_DATA_S(zvec));
  return solver.jacobian_prec(tt, src_yy, src_yp, alpha, rhs, output);
}


template <typename VEC>
DAETimeIntegrator<VEC>::DAETimeIntegrator(OdeArgument<VEC> &bubble) :
  ParameterAcceptor("IDA Solver Parameters"),
  solver(bubble),
  is_initialized(false)
{
  initial_step_size = 1e-4;
  min_step_size = 1e-6;

  abs_tol = 1e-6;
  rel_tol = 1e-8;

  ida_mem = IDACreate();
  is_initialized = true;

}

template <typename VEC>
DAETimeIntegrator<VEC>::~DAETimeIntegrator()
{
  if (ida_mem)
    IDAFree(&ida_mem);
}

template <typename VEC>
void DAETimeIntegrator<VEC>::declare_parameters(ParameterHandler &prm)
{
  add_parameter(prm, &iterative_solver_type,
                "Iterative algorithm", "gmres",
                Patterns::Selection("gmres|bicgs|tfqmr"));

  add_parameter(prm, &provide_jac,
                "Provide jacobian product", "false", Patterns::Bool());

  add_parameter(prm, &provide_jac_prec,
                "Provide jacobian preconditioner", "false", Patterns::Bool());

  add_parameter(prm, &initial_step_size,
                "Initial step size", "1e-4", Patterns::Double());

  add_parameter(prm, &min_step_size,
                "Min step size", "5e-5", Patterns::Double());

  add_parameter(prm, &abs_tol,
                "Absolute error tolerance", "1e-4", Patterns::Double());

  add_parameter(prm, &rel_tol,
                "Relative error tolerance", "1e-3", Patterns::Double());

  add_parameter(prm, &initial_time,
                "Initial time", "0", Patterns::Double());

  add_parameter(prm, &final_time,
                "Final time", "100000", Patterns::Double());

  add_parameter(prm, &outputs_period,
                "Seconds between each output", "1e-1", Patterns::Double());

  add_parameter(prm, &ic_type,
                "Initial condition type", "use_diff_y",
                Patterns::Selection("none|use_diff_y|use_y_dot"),
                "This is one of the following thress options for the "
                "initial condition calculation. \n"
                " none: do not try to make initial conditions consistent. \n"
                " use_diff_y: compute the algebraic components of y and differential\n"
                "    components of y_dot, given the differential components of y. \n"
                "    This option requires that the user specifies differential and \n"
                "    algebraic components in the function get_differential_components.\n"
                " use_y_dot: compute all components of y, given y_dot.");

  add_parameter(prm, &ic_alpha,
                "Initial condition Newton parameter", "0.33", Patterns::Double());


  add_parameter(prm, &ic_max_iter,
                "Initial condition Newton max iterations", "5", Patterns::Integer());

  add_parameter(prm, &use_local_tolerances,
                "Use local tolerances", "false", Patterns::Bool());
}


template <typename VEC>
unsigned int DAETimeIntegrator<VEC>::start_ode(VEC &solution,
                                               VEC &solution_dot,
                                               const unsigned int max_steps)
{


  AssertThrow(solution.size() == solver.n_dofs(),
              ExcDimensionMismatch(solution.size(), solver.n_dofs()));

  AssertThrow(is_initialized, ExcMessage("Not Initialized!"));

  double t = initial_time;
  double h = initial_step_size;
  unsigned int step_number = 0;

  int status;

  // The solution is stored in
  // solution. Here we take only a
  // view of it.

  IndexSet is = solution.locally_owned_elements();

  reset_ode(initial_time, solution, solution_dot, initial_step_size, max_steps);

  copy(yy, solution);
  copy(yp, solution_dot);

  double next_time = 0;

  solver.output_step( 0, solution, solution_dot, 0, initial_step_size);

  while ((t<final_time) && (step_number < max_steps))
    {

      next_time += outputs_period;
      cout << t <<"---->"<<next_time<<endl;
      status = IDASolve(ida_mem, next_time, &t, yy, yp, IDA_NORMAL);

      status = IDAGetLastStep(ida_mem, &h);
      AssertThrow(status == 0, ExcMessage("Error in IDA Solver"));
      cout << "Step " << step_number
           << ", t = " << t
           << ", h = " << h << endl;

      copy(solution, yy);
      copy(solution_dot, yp);

      // Check the solution
      bool reset = solver.solution_check(t, solution, solution_dot, step_number, h);


      solver.output_step(t, solution, solution_dot,  step_number, h);

      if ( reset == true )
        {
          double frac = 0;
          int k = 0;
          IDAGetLastOrder(ida_mem, &k);
          frac = std::pow((double)k,2.);
          reset_ode(t, solution, solution_dot,
                    h/frac, max_steps);
        }


      step_number++;
    }

  // Free the vectors which are no longer used.
  N_VDestroy_Parallel(yy);
  N_VDestroy_Parallel(yp);
  N_VDestroy_Parallel(abs_tolls);
  N_VDestroy_Parallel(diff_id);

  return step_number;
}

template <typename VEC>
void DAETimeIntegrator<VEC>::reset_ode(double current_time,
                                       VEC &solution,
                                       VEC &solution_dot,
                                       double current_time_step,
                                       unsigned int max_steps)
{
  if (ida_mem)
    IDAFree(&ida_mem);

  ida_mem = IDACreate();


  // Free the vectors which are no longer used.
  if (yy)
    {
      N_VDestroy_Parallel(yy);
      N_VDestroy_Parallel(yp);
      N_VDestroy_Parallel(abs_tolls);
      N_VDestroy_Parallel(diff_id);
    }

  int status;
  Assert(solution.size() == solver.n_dofs(),
         ExcDimensionMismatch(solution.size(), solver.n_dofs()));

  Assert(solution_dot.size() == solver.n_dofs(),
         ExcDimensionMismatch(solution_dot.size(), solver.n_dofs()));


  IndexSet is = solution.locally_owned_elements();

  yy        = N_VNew_Parallel(solver.get_comm(), is.n_elements(), solver.n_dofs());
  yp        = N_VNew_Parallel(solver.get_comm(), is.n_elements(), solver.n_dofs());
  diff_id   = N_VNew_Parallel(solver.get_comm(), is.n_elements(), solver.n_dofs());
  abs_tolls = N_VNew_Parallel(solver.get_comm(), is.n_elements(), solver.n_dofs());

  copy(yy, solution);
  copy(yp, solution_dot);
  copy(diff_id, solver.differential_components());

  status = IDAInit(ida_mem, t_dae_residual<VEC>, current_time, yy, yp);

  if (use_local_tolerances)
    {
      VEC &tolerances = solver.get_local_tolerances();
      VEC abs_tolerances(tolerances);
      abs_tolerances /= tolerances.linfty_norm();
      abs_tolerances *= abs_tol;
      copy(abs_tolls, abs_tolerances);
      status += IDASVtolerances(ida_mem, rel_tol, abs_tolls);
    }
  else
    {
      status += IDASStolerances(ida_mem, rel_tol, abs_tol);
    }

  status += IDASetInitStep(ida_mem, current_time_step);
  status += IDASetUserData(ida_mem, (void *) &solver);

  status += IDASetId(ida_mem, diff_id);
  status += IDASetSuppressAlg(ida_mem, TRUE);

  status += IDASetMaxNumSteps(ida_mem, max_steps);
  status += IDASetStopTime(ida_mem, final_time);

  status += IDASetMaxNonlinIters(ida_mem, 10);

  if (iterative_solver_type == "gmres")
    {
      status += IDASpgmr(ida_mem, solver.n_dofs());
    }
  else if (iterative_solver_type == "bicgs")
    {
      status += IDASpbcg(ida_mem, solver.n_dofs());
    }
  else if (iterative_solver_type == "tfqmr")
    {
      status += IDASptfqmr(ida_mem, solver.n_dofs());
    }
  else
    {
      Assert(false, ExcInternalError("I don't know what solver you want to use!"));
    }

  if (provide_jac)
    status += IDASpilsSetJacTimesVecFn(ida_mem, t_dae_jtimes<VEC>);
  if (provide_jac_prec)
    status += IDASpilsSetPreconditioner(ida_mem, t_dae_setup_prec<VEC>, t_dae_prec<VEC>);

  status += IDASetMaxOrd(ida_mem, 5);
  //std::cout<<"???1"<<std::endl;

  AssertThrow(status == 0, ExcMessage("Error initializing IDA."));
  //std::cout<<"???1"<<std::endl;
  if (ic_type == "use_y_dot")
    {
      // (re)initialization of the vectors
      //solution_dot = 0;
      if (current_time !=0)
        IDACalcIC(ida_mem, IDA_Y_INIT, current_time+current_time_step);
      IDAGetConsistentIC(ida_mem, yy, yp);
    }
  else if (ic_type == "use_diff_y")
    {
      IDACalcIC(ida_mem, IDA_YA_YDP_INIT, current_time+current_time_step);
      IDAGetConsistentIC(ida_mem, yy, yp);
    }
  else if (ic_type == "none")
    {
      IDAGetConsistentIC(ida_mem, yy, yp);
    }
  copy(solution, yy);
  copy(solution_dot, yp);

  //    shared_ptr<VEC> resid = solver.create_new_vector();
  //    solver.residual(current_time,*resid,solution,solution_dot);
  //    solution_dot -= *resid;
  //    solver.output_step(solution, solution_dot, 0, 0, current_time_step);

  //solution_dot.reinit(solver.n_dofs());
  //Vector<double> res(solver.n_dofs());
  //solver.residual(0,res,solution_dot,solution);
  //solution_dot -= res;
  //solver.output_step(solution, solution_dot, 0, 0, current_time_step);
}



template <>
unsigned int DAETimeIntegrator<Vector<double> >::start_ode(Vector<double> &solution,
                                                           Vector<double> &solution_dot,
                                                           const unsigned int max_steps)
{


  AssertThrow(solution.size() == solver.n_dofs(),
              ExcDimensionMismatch(solution.size(), solver.n_dofs()));

  AssertThrow(is_initialized, ExcMessage("Not Initialized!"));

  double t = initial_time;
  double h = initial_step_size;
  unsigned int step_number = 0;

  int status;

  // The solution is stored in
  // solution. Here we take only a
  // view of it.

  yy = N_VNewEmpty_Serial(solver.n_dofs());
  NV_DATA_S(yy) = solution.begin();

  yp = N_VNewEmpty_Serial(solver.n_dofs());
  NV_DATA_S(yp) = solution_dot.begin();

  diff_id = N_VNewEmpty_Serial(solver.n_dofs());
  NV_DATA_S(diff_id) = solver.differential_components().begin();

  Vector<double> tolerances = solver.get_local_tolerances();
  tolerances*=(1/tolerances.linfty_norm()*abs_tol);
  abs_tolls = N_VNewEmpty_Serial(solver.n_dofs());
  NV_DATA_S(abs_tolls) = tolerances.begin();

  status = IDAInit(ida_mem, dae_residual, initial_time, yy, yp);
  //status += IDASStolerances(ida_mem, rel_tol, abs_tol);
  status += IDASVtolerances(ida_mem, rel_tol, abs_tolls);
  status += IDASetInitStep(ida_mem, step_number);
  status += IDASetUserData(ida_mem, (void *) &solver);
  //status += IDASetMaxNonlinIters(ida_mem, 60);
  //AssertThrow(status == 0, ExcMessage("Error in IDA Solver"));
  //    status += IDASetNonlinConvCoef(ida_mem, 10.0);
  //status += IDASetMaxOrd(ida_mem, 2);

  reset_ode(initial_time, solution, solution_dot, initial_step_size, max_steps);

  double next_time = 0;
  while ((t<final_time) && (step_number < max_steps))
    {

      next_time += outputs_period;
      cout << t <<"---->"<<next_time<<endl;
      status = IDASolve(ida_mem, next_time, &t, yy, yp, IDA_NORMAL);

      status = IDAGetLastStep(ida_mem, &h);
      AssertThrow(status == 0, ExcMessage("Error in IDA Solver"));
      cout << "Step " << step_number
           << ", t = " << t
           << ", h = " << h << endl;

      // Check the solution
      bool reset = solver.solution_check(t, solution, solution_dot, step_number, h);


      solver.output_step(t, solution, solution_dot, step_number, h);

      if ( reset == true )
        {
          NV_LENGTH_S(yy) = solution.size();
          NV_DATA_S(yy) = solution.begin();
          NV_LENGTH_S(yp) = solution_dot.size();
          NV_DATA_S(yp) = solution_dot.begin();

          double frac = 0;
          int k = 0;
          IDAGetLastOrder(ida_mem, &k);
          frac = std::pow((double)k,2.);
          reset_ode(t, solution, solution_dot,
                    h/frac, max_steps);
        }


      step_number++;
    }

  // Free the vectors which are no longer used.
  N_VDestroy_Serial(yy);
  N_VDestroy_Serial(yp);
  N_VDestroy_Serial(abs_tolls);
  N_VDestroy_Serial(diff_id);

  return step_number;
}

template <>
void DAETimeIntegrator<Vector<double> >::reset_ode(double current_time,
                                                   Vector<double> &solution,
                                                   Vector<double> &solution_dot,                                                   double current_time_step,
                                                   unsigned int max_steps)
{
  if (ida_mem)
    IDAFree(&ida_mem);

  ida_mem = IDACreate();

  int status;
  Assert(solution.size() == solver.n_dofs(),
         ExcDimensionMismatch(solution.size(), solver.n_dofs()));

  Assert(solution_dot.size() == solver.n_dofs(),
         ExcDimensionMismatch(solution_dot.size(), solver.n_dofs()));

  // Free the vectors which are no longer used.
  if (yy)
    N_VDestroy_Serial(yy);
  if (yp)
    N_VDestroy_Serial(yp);
  if (abs_tolls)
    N_VDestroy_Serial(abs_tolls);
  if (diff_id)
    N_VDestroy_Serial(diff_id);


  // The solution is stored in
  // solution. Here we take only a
  // view of it.
  yy = N_VNewEmpty_Serial(solver.n_dofs());
  NV_DATA_S(yy) = solution.begin();

  //N_VPrint_Serial(yy);
  //solution_dot.print();
  yp = N_VNewEmpty_Serial(solver.n_dofs());
  NV_DATA_S(yp) = solution_dot.begin();
  //N_VPrint_Serial(yp);

  diff_id = N_VNewEmpty_Serial(solver.n_dofs());
  NV_DATA_S(diff_id) = solver.differential_components().begin();

  Vector<double> tolerances = solver.get_local_tolerances();
  tolerances*=(1/tolerances.linfty_norm()*abs_tol);
  abs_tolls = N_VNewEmpty_Serial(solver.n_dofs());
  NV_DATA_S(abs_tolls) = tolerances.begin();
  //N_VPrint_Serial(abs_tolls);

  status = IDAInit(ida_mem, dae_residual, current_time, yy, yp);
  //status += IDASStolerances(ida_mem, rel_tol, abs_tol);
  status += IDASVtolerances(ida_mem, rel_tol, abs_tolls);
  status += IDASetInitStep(ida_mem, current_time_step);
  status += IDASetUserData(ida_mem, (void *) &solver);

  status += IDASetId(ida_mem, diff_id);
  status += IDASetSuppressAlg(ida_mem, TRUE);

  status += IDASetMaxNumSteps(ida_mem, max_steps);
  status += IDASetStopTime(ida_mem, final_time);

  status += IDASetMaxNonlinIters(ida_mem, 10);


//    if (use_iterative == true)
//    {
  status += IDASpgmr(ida_mem, solver.n_dofs());
  if (provide_jac)
    status += IDASpilsSetJacTimesVecFn(ida_mem, dae_jtimes);
  if (provide_jac_prec)
    status += IDASpilsSetPreconditioner(ida_mem, dae_setup_prec, dae_prec);
//    }
//    else
//    {
//        status += IDALapackDense(ida_mem, solver.n_dofs());
//    }

  status += IDASetMaxOrd(ida_mem, 5);
  //std::cout<<"???1"<<std::endl;

  AssertThrow(status == 0, ExcMessage("Error initializing IDA."));
  //std::cout<<"???1"<<std::endl;

  status += IDASetNonlinConvCoefIC(ida_mem, ic_alpha);
  status += IDASetMaxNumItersIC(ida_mem, ic_max_iter);
  if (ic_type == "use_diff_y")
    {
      // (re)initialization of the vectors
      //solution_dot = 0;
      if (current_time !=0)
        IDACalcIC(ida_mem, IDA_Y_INIT, current_time+current_time_step);
      IDAGetConsistentIC(ida_mem, yy, yp);
    }
  else if (ic_type == "use_y_dot")
    {
      IDACalcIC(ida_mem, IDA_YA_YDP_INIT, current_time+current_time_step);
      IDAGetConsistentIC(ida_mem, yy, yp);
    }
  else if (ic_type == "none")
    {
      IDAGetConsistentIC(ida_mem, yy, yp);
      std::cout << "Using consistent conditions type 3" << std::endl;
    }
  Vector<double> resid(solver.n_dofs());
  solver.residual(current_time,resid,solution,solution_dot);

  AssertThrow(status == 0, ExcMessage("Error computing IC."));

  //solution_dot.reinit(solver.n_dofs());
  //Vector<double> res(solver.n_dofs());
  //solver.residual(0,res,solution_dot,solution);
  //solution_dot -= res;
  //solver.output_step(solution, solution_dot, 0, 0, current_time_step);
}


template class DAETimeIntegrator<Vector<double> >;
#ifdef DEAL_II_WITH_TRILINOS
template class DAETimeIntegrator<TrilinosWrappers::MPI::BlockVector>;
#endif


