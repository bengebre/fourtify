# Fourtify
Fourtify finds additional sources along a specified orbit in a transient source dataset in order to extend linkages to 4+ nights.

## Installation
While Fourtify is a generalized implementation for any observer at any location, this initial release was designed to work "out of the box" with the Vera C. Rubin observatory's DP0.3 simulation data on the Rubin Science Platform (RSP). As such, the example code in this repository currently requires access to the RSP to run. With that said, the code should be explanatory for those looking to use Fourtify with observations from other observatories.

<br />

Fourtify can be installed like so:
```console
pip install git+https://github.com/bengebre/fourtify
```

## Usage

Fourtify is initialized in the examlpe below via `Fourtify()` with the observational data you will be searching: `obs_radecs`: *observations RA and DEC*, `obs_times`: *observation times*, and `obs_locs`: *observer heliocentric locations*.  You only have to initialize the observational data once.  

Next, the `orbit()` method is used to search along the specified orbit defined by `elems`: *(a,e,i,peri,node,M)* at the `epoch` *time* for observations that are within (in the example below) *10"*  at most and deviating at a rate no more than *1" per day* from this orbit (a cone that spreads out 1"/day until it reaches 10" and is fixed at 10" thereafter).  

The `orbit()` method returns `dradecs`: *the residuals of the sources found in the observation set that meet these criteria* and `fidx`: *the indices of those sources*.

<br />

```python
ff = Fourtify(obs_radecs,obs_times,obs_locs)    #observer RA/DEC (degrees), observation times (TDB jdate), observer locations (heliocentric AU)
dradecs,fidx = ff.orbit(elems,epoch,(10,1))     #(a,e,i,peri,node,M) (degrees), orbit epoch (TDB jdate), (deviation, deviation rate) (arcsec, arcsec/day)
```

## Example notebooks

For a more complete example, the rendered notebooks ```FourtifyExamples.ipynb``` in the [examples/](https://github.com/bengebre/fourtify/blob/main/examples/) directory extends a 3 night candidate link found with HelioLinC in DP0.3 data to 8 nights.  If you haven't calculated the orbit yet, ```FopyFourtifyExamples.ipynb``` shows how to use [Fopy](https://github.com/bengebre/fopy) to call [Find_Orb](https://www.projectpluto.com/find_orb.htm) to get an orbit solution you can pass to Fourtify.

## Acknowlegements

Developer and maintainer:
- [Ben Engebreth](https://benengebreth.org/)

Contributors and collaborators:
- [Siegfried Eggl](https://aerospace.illinois.edu/directory/profile/eggl)
- [Ari Heinze](https://astro.washington.edu/people/aren-heinze)

[DP0.3](https://dp0-3.lsst.io/index.html) is a Data Preview of simulated solar system objects for the [Rubin Observatory's](https://rubinobservatory.org/) [Legacy Survey of Space and Time (LSST)](https://rubinobservatory.org/explore/lsst) created by the [Solar System Science Collaboration](https://lsst-sssc.github.io/).

**Credit**: The DP0.3 data set was generated by members of the Rubin Solar System Pipelines and Commissioning teams, with help from the LSST Solar System Science Collaboration, in particular: Pedro Bernardinelli, Jake Kurlander, Joachim Moeyens, Samuel Cornwall, Ari Heinze, Steph Merritt, Lynne Jones, Siegfried Eggl, Meg Schwamb, Grigori Fedorets, and Mario Juric.
