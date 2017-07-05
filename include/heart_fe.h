#ifndef __heart_fe_h
#define __heart_fe_h

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>

using namespace dealii;

template <int dim,int spacedim>
class Heart
{
public:
  Heart ();
  Heart (bool, const int);
  Point<spacedim> push_forward (const Point<dim>, const int) const;

  void operator()(unsigned int);

private:
  void reinit_data ();
  void setup_system ();
  void run_side ();
  void run_bottom ();

  Triangulation<dim>		   		triangulation;
  FESystem<dim> 			       	fe;
  DoFHandler<dim>			      	dof_handler;
  std::vector<Vector<double> >      solution;

  bool side;
  int heartstep;
};

#endif
