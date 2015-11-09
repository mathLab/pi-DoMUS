#include "interfaces/non_conservative/navier_stokes.h"
#include "pidomus.h"

#include "deal.II/base/numbers.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_Version.hpp"

#include "mpi.h"
#include <iostream>
#include <string>

void print_status(  std::string name,
                    std::string prm_file,
                    int dim,
                    int spacedim,
                    int n_threads,
                    const MPI_Comm &comm);


int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace deal2lkit;

  Teuchos::CommandLineProcessor My_CLP;
  My_CLP.setDocString(
    ".__________        _______   ______   .___  ___.  __    __       _______. \n"
    "|_   __  __|      |       \\ /  __  \\  |   \\/   | |  |  |  |     /       | \n"
    "  | | | |   ______|  .--.  |  |  |  | |  \\  /  | |  |  |  |    |   (----` \n"
    "  | | | |  |______|  |  |  |  |  |  | |  |\\/|  | |  |  |  |     \\   \\     \n"
    "  | | | |         |  '--'  |  `--'  | |  |  |  | |  `--'  | .----)   |    \n"
    "  |_| |_|         |_______/ \\______/  |__|  |__|  \\______/  |_______/     \n\n"
  );

  std::string pde_name="navier_stokes";
  My_CLP.setOption("pde", &pde_name, "name of the PDE (heat, stokes, dynamic_stokes, or navier_stokes)");

  int spacedim = 2;
  My_CLP.setOption("spacedim", &spacedim, "dimensione of the whole space");

  int dim = 2;
  My_CLP.setOption("dim", &dim, "dimension of the problem");

  int n_threads = 0;
  My_CLP.setOption("n_threads", &n_threads, "number of threads");

  std::string prm_file=pde_name+".prm";
  My_CLP.setOption("prm", &prm_file, "name of the parameter file");

  // My_CLP.recogniseAllOptions(true);
  My_CLP.throwExceptions(false);

  Teuchos::CommandLineProcessor::EParseCommandLineReturn
  parseReturn= My_CLP.parse( argc, argv );

  if ( parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED )
    {
      return 0;
    }
  if ( parseReturn != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL   )
    {
      return 1; // Error!
    }

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      n_threads == 0 ? numbers::invalid_unsigned_int  : n_threads);

  const MPI_Comm &comm  = MPI_COMM_WORLD;

  Teuchos::oblackholestream blackhole;
  std::ostream &out = ( Utilities::MPI::this_mpi_process(comm) == 0 ? std::cout : blackhole );

  My_CLP.printHelpMessage(argv[0], out);

  deallog.depth_console (0);

  if (pde_name == "navier_stokes")
    {


      print_status(   "Navier-Stokes Equation",
                      prm_file,
                      dim,
                      spacedim,
                      n_threads,
                      comm);

      if (dim==2)
        {
          NavierStokes<2> energy;
          piDoMUS<2,2,3> navier_stokes_equation (energy);
          ParameterAcceptor::initialize(prm_file, pde_name+"_used.prm");
          ParameterAcceptor::prm.log_parameters(deallog);
          navier_stokes_equation.run ();
        }
      else
        {
          NavierStokes<3> energy;
          piDoMUS<3,3,4> navier_stokes_equation (energy);
          ParameterAcceptor::initialize(prm_file, pde_name+"_used.prm");
          ParameterAcceptor::prm.log_parameters(deallog);
          navier_stokes_equation.run ();
        }
    }
  else
    {
      out << std::endl
          << "============================================================="
          << std::endl
          << " ERROR:"
          << std::endl
          << "  " << pde_name << " needs to be implemented or it is bad name."
          << std::endl
          << "=============================================================";
    }

  out << std::endl;
  return 0;
}

void print_status(  std::string name,
                    std::string prm_file,
                    int dim,
                    int spacedim,
                    const MPI_Comm &comm)
{
  int numprocs  = Utilities::MPI::n_mpi_processes(comm);
  int myid      = Utilities::MPI::this_mpi_process(comm);

  Teuchos::oblackholestream blackhole;
  std::ostream &out = ( Utilities::MPI::this_mpi_process(comm) == 0 ? std::cout : blackhole );

  if (myid == 0)
    {
      out << std::endl
          << "============================================================="
          << std::endl
          << "    Name:   " << name
          // << std::endl
          // << "-------------------------------------------------------------"
          << std::endl
          << " Prm file:  " << prm_file
          << std::endl
          << "n threads:  " <<n_threads
          << std::endl
          << " spacedim:  " << spacedim
          << std::endl
          << "      dim:  " << dim
          << std::endl
          << "    codim:  " << spacedim-dim
          << std::endl
          << "-------------------------------------------------------------"
          << std::endl;
    }
  out << " Process " << getpid() << " is " << myid
      << "   of " << numprocs << " processes" << std::endl;

  if (myid == 0)
    {
      out << "-------------------------------------------------------------"
          << std::endl;
      system("read -p \" Press [Enter] key to start...\"");
      out << "============================================================="
          <<std::endl<<std::endl;
    }
}
