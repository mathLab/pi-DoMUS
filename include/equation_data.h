#ifndef EQUATION_DATA 
#define EQUATION_DATA

using namespace dealii;

namespace EquationData
{
	const double eta                   = 1e21;    /* Pa s       */
	const double density               = 3300;    /* kg / m^3   */  

	const double nu                    = 1.0; 

	template <int dim>
	Tensor<1,dim> gravity_vector (const Point<dim> &p)
	{
		const double r = p.norm();
		return 0*r*p;
	}

}

#endif
