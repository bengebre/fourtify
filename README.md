# Fourtify
Fourtify finds additional transient sources along a specified orbit in a transient source dataset in order to extend linkages to 4+ nights.

## Installation
While Fourtify is a generalized implementation for any observer at any location, this initial release was designed to work "out of the box" with the Vera C. Rubin telescope's DP0.3 simulation data on the Rubin Science Platform (RSP). As such, the example code in this repository currently requires access to the RSP to run. With that said, the code should be explanatory for those looking to use Fourtify with observations from other observatories.

Fourtify can be installed like so:
```console
pip install git+https://github.com/bengebre/fourtify
```

## Example code

The rendered notebook ```FourtifyExamples.ipynb``` in the [examples/](https://github.com/bengebre/fourtify/blob/main/examples/FourtifyExamples.ipynb) directory extends a 3 night candidate link found with HelioLinC in DP0.3 data to 8 nights.

## Acknowlegements

Developer and maintainer:
- [Ben Engebreth](https://benengebreth.org/)

Contributors and collaborators:
- [Siegfried Eggl](https://aerospace.illinois.edu/directory/profile/eggl)
- [Ari Heinze](https://astro.washington.edu/people/aren-heinze)

[DP0.3](https://dp0-3.lsst.io/index.html) is a Data Preview of simulated solar system objects for the [Rubin Observatory's](https://rubinobservatory.org/) [Legacy Survey of Space and Time (LSST)](https://rubinobservatory.org/explore/lsst) created by the [Solar System Science Collaboration](https://lsst-sssc.github.io/).

**Credit**: The DP0.3 data set was generated by members of the Rubin Solar System Pipelines and Commissioning teams, with help from the LSST Solar System Science Collaboration, in particular: Pedro Bernardinelli, Jake Kurlander, Joachim Moeyens, Samuel Cornwall, Ari Heinze, Steph Merritt, Lynne Jones, Siegfried Eggl, Meg Schwamb, Grigori Fedorets, and Mario Juric.
