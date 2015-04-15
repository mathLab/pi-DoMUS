#include "boussinesq_flow_problem.h"

int main (int argc, char *argv[])
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  /*Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);*/

  try
    {
      deallog.depth_console (0);

      std::string parameter_filename;
      if (argc>=2)
        parameter_filename = argv[1];
      else
        parameter_filename = "stokes.prm";

      const int dim = 2;
      BoussinesqFlowProblem<dim>::Parameters  parameters(parameter_filename);
      BoussinesqFlowProblem<dim> flow_problem (parameters, BoussinesqFlowProblem<dim>::global_refinement);
      ParameterAcceptor::initialize("params.prm");
      flow_problem.run ();
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
