#include "boundary_values.h"
#include <cmath>
#define PI 3.14159265358979323846

using namespace dealii;

// - - - - -  public functions - - - - -
template <int dim>
double
BoundaryValues<dim>::value (const Point<dim>  &p,
                            const unsigned int component) const
{
  Assert (component < this->n_components,
          ExcIndexRange (component, 0, this->n_components));

  Vector<double> values(3);
  BoundaryValues<dim>::vector_value (p, values);

  if (component == 0)
      return values(0);
  if (component == 1)
      return values(1);
  if (component == 2)
      return values(2);
  return 0;
}

template <int dim>
void
BoundaryValues<dim>::vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const
{
    if(dt)
        BoundaryValues<dim>::get_values_dt(p, values);
    else
    {
        if (color == 2) {
            get_heartdelta(p, values, heartstep);
        }
        else
            BoundaryValues<dim>::get_values(p, values);
    }
}



// - - - - -  private functions - - - - -

template <int dim>
void
BoundaryValues<dim>::transform_to_polar_coord (const Point<3> &p,
                                               double rot,
                                               double &angle, 
                                               double &height) const
{
  // convert point to polar coordinates
  double x   = p[0],
         y   = p[1],
         z   = p[2],
         phi = atan2(y,z); // returns angle in the range from -Pi to Pi

  // need to rotate to match heart_fe
  phi = phi+rot;

  // angle needs to be in the range from 0 to 2Pi
  phi = (phi < 0) ? phi+2*PI : phi;
  
  angle = phi;
  height = x;
}

template <int dim>
void
BoundaryValues<dim>::swap_coord (Point<3> &p) const
{
  // swap x and z coordinate
  double tmp = p(0);
  p(0) = p(2);
  p(2) = tmp;
}

template <int dim>
void
BoundaryValues<dim>::get_heartdelta (const Point<dim> &p,
                                     Vector<double>   &values,
                                     int heartstep) const
{
  if (color == 2)         //////////////////////////////// top face 
  {
      // convert to polar coordinates and rotate 45 degrees
      double phi, h, rot = PI/4;
      transform_to_polar_coord(p, rot, phi, h);

      // transform back to cartesian
      double x, y, r;
      x = p(2);
      y = p(1);

      r = sqrt(x*x + y*y);
      x = r * cos(phi);
      y = r * sin(phi);
      
      // calc delta
      values(0) = 0;
      values(1) = y - p(1);
      values(2) = x - p(2);
  }
  else if (color == 1)    //////////////////////////////// bottom face
  {
      double x, y;
      x = p(2);
      y = p(1);

      Point<2> two_dim_pnt (x, y);

      //get heart boundary point
      Point<3> heart_p = heart.push_forward (two_dim_pnt, heartstep);
      swap_coord(heart_p);

      // calc delta
      values(0) = heart_p(0) - p(0);
      values(1) = heart_p(1) - p(1);
      values(2) = heart_p(2) - p(2);
  }
  else if (color == 0)    //////////////////////////////// hull
  {
      // convert to polar coordinates and rotate -45 degrees
      double phi, h, rot = -PI/4;
      transform_to_polar_coord(p, rot, phi, h);
      Point<2> polar_pnt (phi, h);
      
      //get heart boundary point
      Point<3> heart_p = heart.push_forward (polar_pnt, heartstep);
      swap_coord(heart_p);

      // calc delta
      values(0) = heart_p(0) - p(0);
      values(1) = heart_p(1) - p(1);
      values(2) = heart_p(2) - p(2);
  }
}

template <int dim>
void
BoundaryValues<dim>::get_values (const Point<dim> &p,
                                 Vector<double>   &values) const
{
    Vector<double> u_(dim);                                 // u_t-1
    Vector<double> u(dim);                                  // u_t
    Vector<double> delta_u(dim);                            // u_t - u_t-1
    double substep = (fmod (timestep, heartinterval) + 1)
                     / (heartinterval / dt);

    BoundaryValues<dim>::get_heartdelta(p, u_, (heartstep-1 < 0) ? 0 : heartstep-1);
    BoundaryValues<dim>::get_heartdelta(p, u, heartstep);

    // calc delta_u
    delta_u = u;
    delta_u -= u_;
    // scale delta_u
    delta_u *= substep;

    u_ += delta_u;

    values(0) = u_(0);
    values(1) = u_(1);
    values(2) = u_(2);
}



template <int dim>
void
BoundaryValues<dim>::get_values_dt (const Point<dim> &p,
                                    Vector<double>   &values) const
{
    Vector<double> u(dim);      // u_t
    Vector<double> u_(dim);     // u_t+1
    double substep = (fmod (timestep, heartinterval) + 1)
                     / (heartinterval / dt);

    BoundaryValues<dim>::get_heartdelta(p, u, heartstep);
    BoundaryValues<dim>::get_heartdelta(p, u_, (heartstep+1 > 99) ? 0 : heartstep+1);

    // (u_t+1 - u_t) / h
    u_ -= u;
    u_ /= heartinterval;
    // scale u_
    u_ *= substep;

    values(0) = u_(0);
    values(1) = u_(1);
    values(2) = u_(2);
}

// Explicit instantiations
template class BoundaryValues<3>;
