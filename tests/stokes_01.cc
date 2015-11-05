#include "pidomus.h"
#include "interfaces/conservative/stokes.h"
#include "tests.h"

/*
 * Test the stokes interface with an exact solution.
 */

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  const int dim = 2;
  const int spacedim = 2;

  Stokes<dim> energy;
  piDoMUS<dim,spacedim,dim+1> stokes_flow (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/stokes_01.prm", "used_parameters.prm");

  stokes_flow.run ();

  auto sol = stokes_flow.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << sol[i] << std::endl ;
    }
  return 0;
}
