#ifndef EQUATION_DATA 
#define EQUATION_DATA

using namespace dealii;

namespace EquationData
{
	const double eta                   = 1e21;    /* Pa s       */
	const double kappa                 = 1e-6;    /* m^2 / s    */
	const double density               = 3300;    /* kg / m^3   */  
	const double reference_temperature = 293;     /* K          */
	const double expansion_coefficient = 2e-5;    /* 1/K        */
	const double specific_heat         = 1250;    /* J / K / kg */
	const double radiogenic_heating    = 7.4e-12; /* W / kg     */

	const double R0      = 6371000.-2890000.;     /* m          */
	const double R1      = 6371000.-  35000.;     /* m          */
	const double T0      = 4000+273;              /* K          */
	const double T1      =  700+273;              /* K          */

	// Added in review
	const double temperature_initial_values = 1.0 ;
	const double nu                         = 1.0; 

	template <int dim>
	Tensor<1,dim> gravity_vector (const Point<dim> &p)
	{
		const double r = p.norm();
		return 0*r*p;
	}

	const double pressure_scaling = eta / 10000;
	const double year_in_seconds  = 60*60*24*365.2425;
}

#endif