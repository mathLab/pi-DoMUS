#include "pidomus.h"
#include "interfaces/non_conservative/dynamic_stokes.h"
#include "tests.h"

/*
 * Test the dynamic stokes (using a non conservative interface) interface
 * with an exact solution.
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

  DynamicStokesNC<dim> energy;
  piDoMUS<dim,spacedim,dim+1> dynamic_stokes (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/dynamic_stokes_nc_01.prm", "used_parameters.prm");

  dynamic_stokes.run ();

  auto sol = dynamic_stokes.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << sol[i] << std::endl ;
    }
  return 0;
}
