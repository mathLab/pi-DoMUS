#include "solution.h"


template <int dim>
void Solution<dim>::vector_value (	const Point<dim>   &p,
									Vector<double> &value) const
							// const unsigned int) const
{
	// Tensor<1,dim+1> return_value;
	// Vector<double> value (dim +1);
	value(0) = numbers::PI*cos(numbers::PI*p[0])*cos(numbers::PI*p[1]);
	value(1) = numbers::PI*sin(numbers::PI*p[0])*sin(numbers::PI*p[1]);
	value(2) = 0;
	// if (dim == 3)
	// 	value(3) = 0;
	// 
	// return return_value;
}


template <int dim>
void Solution<dim>::vector_gradient (	const Point<dim>   &p,
										std::vector<Tensor<1,dim>> &gradients) const

{
	// Tensor<2,dim+1> return_value;
	// std::vector<Tensor<1,dim>> gradients (dim + 1);
	
	gradients[0][0] = - numbers::PI*numbers::PI*sin(numbers::PI*p[0])*cos(numbers::PI*p[1]);
	gradients[0][1] = - numbers::PI*numbers::PI*cos(numbers::PI*p[0])*sin(numbers::PI*p[1]);
	gradients[1][0] = numbers::PI*numbers::PI*cos(numbers::PI*p[0])*sin(numbers::PI*p[1]);
	gradients[1][1] = numbers::PI*numbers::PI*sin(numbers::PI*p[0])*cos(numbers::PI*p[1]);
	
}
// 
// template <int dim>
// Tensor<2,dim+1> Solution<dim>::gradient (const Point<dim>   &p,
// 										const unsigned int) const
// {
// 	Tensor<2,dim+1> return_value;
// 	return_value[0][0] = - numbers::PI*numbers::PI*sin(numbers::PI*p[0])*cos(numbers::PI*p[1]);
// 	return_value[0][1] = - numbers::PI*numbers::PI*cos(numbers::PI*p[0])*sin(numbers::PI*p[1]);
// 	return_value[1][0] = numbers::PI*numbers::PI*cos(numbers::PI*p[0])*sin(numbers::PI*p[1]);
// 	return_value[1][1] = numbers::PI*numbers::PI*sin(numbers::PI*p[0])*cos(numbers::PI*p[1]);
// 	
// 	// return_value[1][1] =
// 	// return_value[1][1] =
// 	// return_value[1][1] =
// 	// return_value[1][1] =
// 	// return_value[1][1] =
// 	// return_value[1][1] =
// 	// return_value[1][1] =
// 	// return_value[1][1] =
// 	// 
// 	// if( dim == 3 )
// 	// 	return_value[][1] = 0;
// 	// for (unsigned int i=0; i<this->n_source_centers; ++i)
// 	//   {
// 	//     const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
// 	// 
// 	//     return_value += (-2 / (this->width * this->width) *
// 	//                      std::exp(-x_minus_xi.norm_square() /
// 	//                               (this->width * this->width)) *
// 	//                      x_minus_xi);
// 	//   }
// 
// 	return return_value;
// }

// template class Solution<1>; // DA ELIMINARE!
template class Solution<2>;
// template class Solution<3>;