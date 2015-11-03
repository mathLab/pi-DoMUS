#include "pidomus.h"
#include "interfaces/stokes.h"
#include "interfaces/stokes_nc.h"
#include "lac_type.h"
#include "tests.h"

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
    StokesNC<dim> energy_nc;
    piDoMUS<dim,spacedim,dim+1> stokes_flow_nc (energy_nc);
    ParameterAcceptor::initialize(SOURCE_DIR "/parameters/stokes_nc_01_nc.prm", "used_parameters.prm");
    stokes_flow_nc.run ();
    sol_nc = stokes_flow_nc.get_solution();
  }
  {
    Stokes<dim> energy_c;
    piDoMUS<dim,spacedim,dim+1> stokes_flow_c (energy_c);
    ParameterAcceptor::initialize(SOURCE_DIR "/parameters/stokes_nc_01_c.prm", "used_parameters.prm");
    stokes_flow_c.run ();
    sol_c = stokes_flow_c.get_solution();
  }


  for (unsigned int i = 0 ; i<sol_c.size(); ++i)
    {
      if (std::abs(sol_c[i]-sol_nc[i]) > 1e-2)
        {
          deallog << "FAIL" << std::endl;
        }
    }
  deallog << "DONE!" << std::endl ;
  return 0;
}
