# ULTRASAT-utils
Useful tools for planning ULTRASAT observations

### Installation
```
pip install ultrasatutils
```
Or to install the dev version, clone the repo and use ```poetry install```

### Example useage
```python
import ultrasatutils as usu
usu.magnitude_to_ULTRASAT(teff=6000, logg=4.35, vmag=12)

16.199554882344437
```


```python
from ultrasatutils import get_transit_SNR
from astropy import units as u
usu.get_transit_SNR(mag=16.2, duration=2 * u.hour, depth=0.01)

7.3371085
```
