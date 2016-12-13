#ifndef __boundary_values_h
#define __boundary_values_h

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include "heart_fe.h"
#include <cmath>

using namespace dealii;

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues (int color, bool derivative=false) 
    : 
    Function<dim>(2*dim+1), 
    color((dim==2)?color-1:color),
    derivative(derivative),
    fe(FE_Q<1>(2), dim)
    {}
  BoundaryValues (int color, 
                  double timestep, 
                  double dt, 
                  bool side, 
                  int degree, 
                  bool derivative=false) 
    : 
    Function<dim>(2*dim+1),
    color((dim==2)?color-1:color), 
    timestep(timestep),
    dt(dt),
    heartstep(timestep/heartinterval),
    derivative(derivative),
    heart(side, degree, heartstep),
    fe(FE_Q<1>(2), dim),
    dof_handler(triangulation)
  {
    setup_system();
  }

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &value) const;
private:
  int color;
  double timestep;
  double dt;
  double heartinterval = 0.005;
  int heartstep;// = timestep / heartinterval;
  bool derivative;
  Heart<2,3> heart; 
  Triangulation<1> triangulation;
  FESystem<1>      fe;
  DoFHandler<1>    dof_handler;
  Point<3> rotate (const Point<3> &p, 
                   double rot, 
                   double &angle, 
                   double &height) const;
  void swap_coord(Point<3> &p) const;
  void setup_system();
  void get_heartdelta (const Point<dim> p, Vector<double> &value, int heartstep) const;
  void get_values (const Point<dim> &p, Vector<double> &value) const;
  void get_values_dt (const Point<dim> &p, Vector<double> &value) const;
};

#endif
