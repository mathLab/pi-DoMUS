#include "../include/ode_argument.h"
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>

template<typename VEC>
int OdeArgument<VEC>::setup_jacobian_prec(double,
                                          VEC const &,
                                          VEC const &,
                                          double)
{
  Assert(false, ExcPureFunctionCalled());
  return 0;
}

template<typename VEC>
int OdeArgument<VEC>::jacobian_prec(double, VEC &,
                                    VEC const &,
                                    VEC const &,
                                    VEC const &,
                                    double)
{
  Assert(false, ExcPureFunctionCalled());
  return 0;
}

template<typename VEC>
int OdeArgument<VEC>::jacobian(double,
                               VEC &,
                               VEC const &,
                               VEC const &,
                               VEC const &,
                               double)
{
  Assert(false, ExcPureFunctionCalled());
  return 0;
}

template<typename VEC>
VEC &OdeArgument<VEC>::differential_components()
{
  Assert(false, ExcPureFunctionCalled());
  static VEC tmp;
  return tmp;
}

template<typename VEC>
VEC &OdeArgument<VEC>::get_local_tolerances()
{
  Assert(false, ExcPureFunctionCalled());
  static VEC tmp;
  return tmp;
}

template class OdeArgument<Vector<double> >;
#ifdef DEAL_II_WITH_TRILINOS
template class OdeArgument<TrilinosWrappers::MPI::BlockVector>;
#endif