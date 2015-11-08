#include "interfaces/non_conservative/navier_stokes.h"
#include "pidomus.h"
#include "mpi.h"

void print_status(  std::string name,
                    std::string prm_file,
                    const MPI_Comm &comm);

int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace deal2lkit;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  const MPI_Comm &comm = MPI_COMM_WORLD;

  /*Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);*/

  try
    {
      deallog.depth_console (0);
      //initlog();

      std::string parameter_filename;
      if (argc>=2)
        parameter_filename = argv[1];
      else
        parameter_filename = "../utilities/prm/navier_stokes_default.prm";

      print_status("Navier-Stokes Equation", parameter_filename, comm);

      const int dim = 2;
      const int spacedim = 2;

      NavierStokes<dim> energy;
      piDoMUS<dim,spacedim,dim+1> navier_stokes_equation (energy);
      ParameterAcceptor::initialize(parameter_filename, "used_parameters.prm");

      ParameterAcceptor::prm.log_parameters(deallog);

      navier_stokes_equation.run ();

      std::cout << std::endl;

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "-------------------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "-------------------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "-------------------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "-------------------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

void print_status(  std::string name,
                    std::string prm_file,
                    const MPI_Comm &comm)
{
  int numprocs = Utilities::MPI::n_mpi_processes(comm);
  int myid = Utilities::MPI::this_mpi_process(comm);


  if (myid == 0)
    {
      std::cout << std::endl
                << "============================================================="
                << std::endl
                << "    Name:   " << name
                // << std::endl
                // << "-------------------------------------------------------------"
                << std::endl
                << " Prm file:  " << prm_file
                << std::endl
                << "-------------------------------------------------------------"
                << std::endl;
    }
  std::cout << " Process " << getpid() << " is " << myid
            << "   of " << numprocs << " processes" << std::endl;

  if (myid == 0)
    {
      std::cout << "-------------------------------------------------------------"
                << std::endl;
      system("read -p \" Press [Enter] key to start...\"");
      std::cout << "============================================================="
                <<std::endl<<std::endl;
    }
}
