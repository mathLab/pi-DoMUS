#ifndef __boundary_values_h
#define __boundary_values_h

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include "heart_fe.h"

using namespace dealii;

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues (int cl, bool derivative=false) 
    : 
    Function<dim>(2*dim+1), 
    color(cl),
    dt(derivative)
    {}
  BoundaryValues (int cl, int ts, bool side, int degree, bool derivative=false) 
    : 
    Function<dim>(2*dim+1),
    color(cl), 
    timestep(ts),
    dt(derivative),
    heart(side, degree)
  {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &value) const;
private:
  int color;
  int timestep;
  bool dt;
  Heart<2,3> heart; 
  void transform_to_polar_coord(const Point<3> &p, 
                                double rot, 
                                double &angle, 
                                double &height) const;
  void swap_coord(Point<3> &p) const;
  void get_values (const Point<dim> &p, Vector<double> &value, int timestep) const;
  void get_values_dt (const Point<dim> &p, Vector<double> &value) const;
};

#endif
