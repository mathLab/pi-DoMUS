#include "pidomus.h"
#include "interfaces/conservative/dynamic_stokes.h"
#include "interfaces/non_conservative/dynamic_stokes.h"
#include "lac/lac_type.h"
#include "tests.h"

/*
 * Test the implementation of the dynamic stokes problem using the conservative
 * and the non conservative formulation. This code finds a solution using both
 * codes and then compares the solutions.
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
  LATrilinos::VectorType sol_nc,sol_c;
  {
    DynamicStokesNC<dim> energy_nc;
    piDoMUS<dim,spacedim,dim+1> dynamic_stokes_nc (energy_nc);
    ParameterAcceptor::initialize(SOURCE_DIR "/parameters/dynamic_stokes_nc_02_nc.prm", "used_parameters.prm");
    dynamic_stokes_nc.run ();
    sol_nc = dynamic_stokes_nc.get_solution();
  }
  {
    DynamicStokes<dim> energy_c;
    piDoMUS<dim,spacedim,dim+1> dynamic_stokes (energy_c);
    ParameterAcceptor::initialize(SOURCE_DIR "/parameters/dynamic_stokes_nc_02_c.prm", "used_parameters.prm");
    dynamic_stokes.run ();
    sol_c = dynamic_stokes.get_solution();
  }


  for (unsigned int i = 0 ; i<sol_c.size(); ++i)
    {
      if (std::abs(sol_c[i]-sol_nc[i]) > 1e-4)
        {
          deallog << "FAIL" << std::endl;
        }
    }
  deallog << "DONE!" << std::endl ;
  return 0;
}
