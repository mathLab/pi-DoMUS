#include "pidomus.h"
#include "interfaces/poisson_problem.h"
#include "tests.h"

#include <algorithm>


// helper function for sorting the eigenvalues
bool myfunction(std::complex<double> i, std::complex<double> j)
{
  double a = i.real();
  double b = j.real();

  return (a<b);
}

using namespace dealii;

int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  const int dim = 2;
  const int spacedim = 2;

  PoissonProblem<dim,spacedim,LADealII> p;
  piDoMUS<dim,spacedim,LADealII> solver ("pidomus",p);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/arpack_01.prm", "used_parameters.prm");


  solver.solve_eigenproblem ();

  auto eigenvalues = solver.get_eigenvalues();

  std::sort(eigenvalues.begin(), eigenvalues.end(), myfunction);

  for (unsigned int i=0; i< eigenvalues.size(); ++ i)
    deallog << eigenvalues[i].real()/(numbers::PI*numbers::PI)
            << std::endl;

  return 0;
}
