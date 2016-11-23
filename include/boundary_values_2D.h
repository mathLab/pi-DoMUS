#ifndef __boundary_values_h
#define __boundary_values_h

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
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
    color(color),
    derivative(derivative)
    {}
  BoundaryValues (int color, 
                  double timestep, 
                  double dt, 
                  bool side, 
                  int degree, 
                  bool derivative=false) 
    : 
    Function<dim>(2*dim+1),
    color(color), 
    timestep(timestep),
    dt(dt),
    heartstep(timestep/heartinterval),
    derivative(derivative),
    heart(side, degree)
  {}

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
  void transform_to_polar_coord(const Point<3> &p, 
                                double rot, 
                                double &angle, 
                                double &height) const;
  void swap_coord(Point<3> &p) const;
  void get_heartdelta (const Point<dim> &p, Vector<double> &value, int heartstep) const;
  void get_values (const Point<dim> &p, Vector<double> &value) const;
  void get_values_dt (const Point<dim> &p, Vector<double> &value) const;
};

#endif
