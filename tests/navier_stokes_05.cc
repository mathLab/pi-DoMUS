#include "pidomus.h"
#include "interfaces/navier_stokes.h"
#include "tests.h"

/**
 * Test:     Navier Stokes interface.
 * Method:   Euler - Approximation of (grad u)u made with grad_linear
 * Problem:  (\nabla u)u + \nabla p = f
 * Exact solution:
 * \f[
 *    u=(cos(x)*cos(y), sin(x)*sin(y) \textrm{ and } p=0;
 * \f]
 */

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  NavierStokes<2,2> energy(true);
  piDoMUS<2,2> navier_stokes ("",energy);
  ParameterAcceptor::initialize(
    SOURCE_DIR "/parameters/navier_stokes_05.prm",
    "used_parameters.prm");

  navier_stokes.run ();

  auto &sol = navier_stokes.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << sol[i] << std::endl ;
    }

  return 0;
}
