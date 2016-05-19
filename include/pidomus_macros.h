#ifndef _pidomus_macros_h
#define _pidomus_macros_h

/**
  * A macro used for instantiating pi-DoMUS classes
  * and functions.
  */
#define PIDOMUS_INSTANTIATE(INSTANTIATIONS) \
  INSTANTIATIONS(2,2,LATrilinos) \
  INSTANTIATIONS(2,3,LATrilinos) \
  INSTANTIATIONS(3,3,LATrilinos) \
  INSTANTIATIONS(2,2,LADealII) \
  INSTANTIATIONS(2,3,LADealII) \
  INSTANTIATIONS(3,3,LADealII)


#endif
