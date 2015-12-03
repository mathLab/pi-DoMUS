#include "pidomus.h"
#include "interfaces/navier_stokes.h"
#include "tests.h"

/**
 * Test:     Navier Stokes interface.
 * Method:   Direct
 * Problem:  Stokes
 * Exact solution:
 * \f[
 *    u=(1, 1\) \textrm{ and } p=0;
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

  NavierStokes<2,2,LADealII> energy(false, false);
  piDoMUS<2,2,LADealII> stokes ("",energy);
  ParameterAcceptor::initialize(
    SOURCE_DIR "/parameters/stokes_00.prm",
    "used_parameters.prm");

  stokes.run ();

  auto sol = stokes.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << sol[i] << std::endl ;
    }

  return 0;
}
