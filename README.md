# MTProps

Calculate local pitch lengths, protofilament numbers using tomograms.

:warning: This package is fully dependent on [impy](https://github.com/hanjinliu/impy).

```python
from mtprops import start
import impy as ip

path = ".../XXX.mrc"
mtprof = start(ip.gui.viewer, path)
```