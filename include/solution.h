#ifndef _SOLUTION_ 
#define _SOLUTION_

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

template <int dim>
class Solution : public Function<dim>
{
	public:
	Solution () : Function<dim>(dim+1) {}

	virtual void value (const Point<dim>   &p,
							Vector<double> &vector_value) const;
							// const unsigned int  component = 0) const;

	/*virtual Tensor<1,dim> value (const Point<dim>   &p,
								const unsigned int  component = 0) const;*/

	// virtual Tensor<2,dim + 1> gradient (const Point<dim>   &p,
									// const unsigned int  component = 0) const;
};

#endif