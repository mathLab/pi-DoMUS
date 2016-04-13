#include "interfaces/hydrogels_one_field.h"
#include "interfaces/hydrogels_two_fields_transient.h"
#include "interfaces/hydrogels_three_fields.h"
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

void print_status(std::string name,
                  std::string prm_file,
                  int dim,
                  int spacedim,
                  int n_threads,
                  const MPI_Comm &comm,
                  bool check_prm);
constexpr int get_constexpr(int n)
{
  return n;
}

template <int dim, int spacedim, typename LAC>
void run(const std::string &pde_name,
         const std::string &prm_file)
{
  if (pde_name == "one-field")
    {
      HydroGelOneField<dim,spacedim,LAC> gel;
      piDoMUS<dim,spacedim,LAC> solver("piDomus", gel);

      ParameterAcceptor::initialize(prm_file, pde_name+"_used.prm");
      ParameterAcceptor::prm.log_parameters(deallog);
      solver.run ();
    }
}

int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace deal2lkit;

  Teuchos::CommandLineProcessor My_CLP;
  My_CLP.setDocString(
    "                ______     ___  ____   _ __________ \n"
    "                |  _  \\    |  \\/  | | | /  ______  \\ \n"
    "  ______________| | | |____| .  . | |_| \\ `--.___| | \n"
    " |__  __  ______| | | / _ \\| |\\/| | |_| |`--. \\____/ \n"
    "   | | | |      | |/ | (_) | |  | | |_| /\\__/ / \n"
    "   |_| |_|      |___/ \\___/\\_|  |_/\\___/\\____/ \n"
  );

  std::string pde_name="three-fields";
  My_CLP.setOption("pde", &pde_name, "name of the PDE (one-field, two-fields or three-fields)");

  std::string LAC = "LADealII";
  My_CLP.setOption("LAC", &LAC, "LADealII|LATrilinos");

  int spacedim = 3;
  My_CLP.setOption("spacedim", &spacedim, "dimensione of the whole space");

  int dim = 3;
  My_CLP.setOption("dim", &dim, "dimension of the problem");

  int n_threads = 0;
  My_CLP.setOption("n_threads", &n_threads, "number of threads");

  std::string prm_file=pde_name+".prm";
  My_CLP.setOption("prm", &prm_file, "name of the parameter file");

  bool check_prm = false;
  My_CLP.setOption("check","no-check", &check_prm, "wait for a key press before starting the run");

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

  // deallog.depth_console (0);
  try
    {
      print_status(   "Solving "+pde_name+" Equations",
                      prm_file,
                      dim,
                      spacedim,
                      n_threads,
                      comm,
                      check_prm);
      if (LAC == "LADealII")
        {
          if (pde_name == "one-field")
            {
              HydroGelOneField<3,3,LADealII> gel;
              piDoMUS<3,3,LADealII> solver ("piDoMUS", gel);
              ParameterAcceptor::initialize(prm_file, pde_name+"_used.prm");
              ParameterAcceptor::prm.log_parameters(deallog);
              solver.run ();
            }
          else if (pde_name == "two-fields")
            {
              HydroGelTwoFieldsTransient<3,3,LADealII> gel;
              piDoMUS<3,3,LADealII> solver ("piDoMUS", gel);
              ParameterAcceptor::initialize(prm_file, pde_name+"_used.prm");
              ParameterAcceptor::prm.log_parameters(deallog);
              solver.run ();
            }
          else if (pde_name == "three-fields")
            {
              HydroGelThreeFields<3,3,LADealII> gel;
              piDoMUS<3,3,LADealII> solver ("piDoMUS", gel);
              ParameterAcceptor::initialize(prm_file, pde_name+"_used.prm");
              ParameterAcceptor::prm.log_parameters(deallog);
              solver.run ();
            }
          else
            {
              AssertThrow(false, ExcMessage(pde_name+" is not known."));
            }

        }
      else if (LAC == "LATrilinos")
        {
          if (pde_name == "one-field")
            {
              HydroGelOneField<3,3,LATrilinos> gel;
              piDoMUS<3,3,LATrilinos> solver ("piDoMUS", gel);
              ParameterAcceptor::initialize(prm_file, pde_name+"_used.prm");
              ParameterAcceptor::prm.log_parameters(deallog);
              solver.run ();
            }
          else if (pde_name == "two-fields")
            {
              HydroGelTwoFieldsTransient<3,3,LATrilinos> gel;
              piDoMUS<3,3,LATrilinos> solver ("piDoMUS", gel);
              ParameterAcceptor::initialize(prm_file, pde_name+"_used.prm");
              ParameterAcceptor::prm.log_parameters(deallog);
              solver.run ();
            }
          else if (pde_name == "three-fields")
            {
              HydroGelThreeFields<3,3,LATrilinos> gel;
              piDoMUS<3,3,LATrilinos> solver ("piDoMUS", gel);
              ParameterAcceptor::initialize(prm_file, pde_name+"_used.prm");
              ParameterAcceptor::prm.log_parameters(deallog);
              solver.run ();
            }
          else
            {
              AssertThrow(false, ExcMessage(pde_name+" is not known."));
            }
        }
      else
        {
          AssertThrow(false, ExcMessage("Linear algebra type named "+LAC+" is not supported."));
        }

      out << std::endl;
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

void print_status(std::string name,
                  std::string prm_file,
                  int dim,
                  int spacedim,
                  int n_threads,
                  const MPI_Comm &comm,
                  bool check_prm)
{
  int numprocs  = Utilities::MPI::n_mpi_processes(comm);
  //  int myid      = Utilities::MPI::this_mpi_process(comm);

  Teuchos::oblackholestream blackhole;
  std::ostream &out = ( Utilities::MPI::this_mpi_process(comm) == 0 ? std::cout : blackhole );


  out << std::endl
      << "============================================================="
      << std::endl
      << "     Name:  " << name
      << std::endl
      << " Prm file:  " << prm_file
      << std::endl
      << "n threads:  " << n_threads
      << std::endl
      << "  process:  "  << getpid()
      << std::endl
      << " proc.tot:  "  << numprocs
      << std::endl
      << " spacedim:  " << spacedim
      << std::endl
      << "      dim:  " << dim
      // << std::endl
      // << "    codim:  " << spacedim-dim
      << std::endl;
  if (check_prm)
    {
      out<< "-------------------------------------------------------------"
         << std::endl;
      out << "Press [Enter] key to start...";
      if (std::cin.get() == '\n') {};
    }
  out << "============================================================="
      <<std::endl<<std::endl;

}
