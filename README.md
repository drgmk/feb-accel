# Exocomet Orbit Fitting: Accelerating Coma Absorption During Transits of beta Pictoris

http://adsabs.harvard.edu/abs/2018MNRAS.479.1997K

Here are files and code used in the production of this paper. Functions that do things like model radial velocity are in the file ```funcs.py```, and higher level things like spectrum extraction, absorption feature fitting, and plotting are in the iPython notebooks.

Output from the fitting is in the ```fits``` directory, but without the stored chains as these files are large. The final steps in the MCMC chains and the corresponding dates are in the files ```eccentric_final.npy```, ```parabolic_final.npy```, and ```dates_final.npy```.
