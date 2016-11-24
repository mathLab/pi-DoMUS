#include "pidomus.h"
#include "interfaces/ALE_navier_stokes.h"
#include "tests.h"

/**
 * Test:     ALE Navier Stokes interface.
 */

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);
  deallog.threshold_double(1.0e-3);

  ALENavierStokes<2,2, LADealII> energy;
  piDoMUS<2,2, LADealII> navier_stokes ("",energy);
  ParameterAcceptor::initialize(
    SOURCE_DIR "/parameters/ALE_navier_stokes_00.prm",
    "used_parameters.prm");

  navier_stokes.run ();

  auto sol = navier_stokes.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << sol[i] << std::endl ;
    }

  return 0;
}
