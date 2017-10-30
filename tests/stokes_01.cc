#include "pidomus.h"
#include "interfaces/stokes.h"
#include "tests.h"


/**
 * Test:     Stokes interface.
 * Method:   Iterative - Euler
 * Problem:  Stokes
 * Exact solution:
 * \f[
 *    u=(x, -y\) \textrm{ and } p=x*y;
 * \f]
 */

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  StokesInterface<2,2,LATrilinos> interface;
  piDoMUS<2,2,LATrilinos> stokes ("pi-DoMUS",interface);
  ParameterAcceptor::initialize(
    SOURCE_DIR "/parameters/stokes_01.prm",
    "used_parameters.prm");

  stokes.run ();

  auto &sol = stokes.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << sol[i] << std::endl ;
    }

  return 0;
}
