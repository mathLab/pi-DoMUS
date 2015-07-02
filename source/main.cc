#include "interfaces/dynamic_stokes_derived_interface.h"
#include "interfaces/heat_equation.h"
#include "n_fields_problem.h"
#include "mpi.h"

int main (int argc, char *argv[])
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  const MPI_Comm &comm = MPI_COMM_WORLD;

  int numprocs = Utilities::MPI::n_mpi_processes(comm);
  int myid = Utilities::MPI::this_mpi_process(comm);


  std::cout << "Process " << getpid() << " is " << myid
            << " of " << numprocs << " processes" << std::endl;
  if (myid == 0) system("read -p \"Press [Enter] key to start debug...\"");


  /*Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);*/

  try
    {
      deallog.depth_console (0);
      //initlog();

      std::string parameter_filename;
      if (argc>=2)
        parameter_filename = argv[1];
      else
        parameter_filename = "parameters.prm";


      const int dim = 2;
      // BoussinesqFlowProblem<dim>::Parameters  parameters(parameter_filename);

      HeatEquation<dim> energy;
      NFieldsProblem<dim,dim,1> n_problem (energy);
      // NavierStokes<dim> flow_problem (NavierStokes<dim>::global_refinement);
      ParameterAcceptor::initialize(parameter_filename, "used_parameters.prm");


      // ParameterAcceptor::initialize("params.prm");
      //ParameterAcceptor::clear();
      ParameterAcceptor::prm.log_parameters(deallog);

      n_problem.run ();

      std::cout << std::endl;

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
