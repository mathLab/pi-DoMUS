#include "pidomus.h"
#include "interfaces/conservative/stokes.h"
#include "lac/lac_type.h"
#include "tests.h"

/*
 * Test two different implementation of the Stokes preconditioner.
 * The first uses block_back_substitution while the other not.
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
  LATrilinos::VectorType sol_bbs_on,sol_bbs_off;
  {
    Stokes<dim> energy_bbs_on;
    piDoMUS<dim,spacedim,dim+1> stokes_flow_bbs_on (energy_bbs_on);
    ParameterAcceptor::initialize(SOURCE_DIR "/parameters/stokes_02_bbs_on.prm", "used_parameters.prm");
    stokes_flow_bbs_on.run ();
    sol_bbs_on = stokes_flow_bbs_on.get_solution();
  }
  {
    Stokes<dim> energy_bbs_off;
    piDoMUS<dim,spacedim,dim+1> stokes_flow_bbs_off (energy_bbs_off);
    ParameterAcceptor::initialize(SOURCE_DIR "/parameters/stokes_02_bbs_off.prm", "used_parameters.prm");
    stokes_flow_bbs_off.run ();
    sol_bbs_off = stokes_flow_bbs_off.get_solution();
  }


  for (unsigned int i = 0 ; i<sol_bbs_off.size(); ++i)
    {
      if (std::abs(sol_bbs_off[i]-sol_bbs_on[i]) > 1e-2)
        {
          deallog << "FAIL" << std::endl;
        }
    }
  deallog << "DONE!" << std::endl ;
  return 0;
}
