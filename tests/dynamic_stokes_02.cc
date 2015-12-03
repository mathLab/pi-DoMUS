#include "pidomus.h"
#include "interfaces/navier_stokes.h"
#include "tests.h"

/**
 * Test:     Navier Stokes interface.
 * Method:   Iterative - Euler
 * Problem:  Dynamic Stokes
 * Exact solution:
 * \f[
 *    u=\big(\cos(x)\cos(y), \sin(x)\sin(y)\big)
 *    \textrm{ and }p=0;
 * \f]
 */

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);
  deallog.threshold_double(1.0e-3);

  NavierStokes<2,2> energy(true, false);
  piDoMUS<2,2> dynamic_stokes ("",energy);
  ParameterAcceptor::initialize(
    SOURCE_DIR "/parameters/dynamic_stokes_01.prm",
    "used_parameters.prm");

  dynamic_stokes.run ();

  auto sol = dynamic_stokes.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << sol[i] << std::endl ;
    }

  return 0;
}
