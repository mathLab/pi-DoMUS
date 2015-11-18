#include "pidomus.h"
#include "interfaces/non_conservative/navier_stokes_aux.h"
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

  NavierStokes<dim> ns;
  piDoMUS<dim,spacedim,dim+1> solver (ns);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/aux_navier_stokes_01.prm", "used_parameters.prm");

  solver.run ();

  auto sol = solver.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << sol[i] << std::endl ;
    }
  return 0;
}
