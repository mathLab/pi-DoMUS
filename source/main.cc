#include "navier_stokes.h"

int main (int argc, char *argv[])
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

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
      
      NavierStokes<dim> flow_problem (NavierStokes<dim>::global_refinement);
      ParameterAcceptor::initialize(parameter_filename, "used_parameters.prm");


      // ParameterAcceptor::initialize("params.prm");
      //ParameterAcceptor::clear();
      ParameterAcceptor::prm.log_parameters(deallog);

      flow_problem.run ();

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
