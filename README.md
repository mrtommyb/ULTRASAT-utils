# ULTRASAT-utils
Useful tools for planning ULTRASAT observations

```python
from ultrasatutils import magnitude_to_ULTRASAT
magnitude_to_ULTRASAT(6000, vmag=12)

16.199554882344437
```


```python
from ultrasatutils import get_transit_SNR
from astropy import units as u
get_transit_SNR(16.2, 2 * u.hour, 0.01)

7.3371085
```
