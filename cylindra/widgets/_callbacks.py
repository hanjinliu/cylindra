from __future__ import annotations

import numpy as np
from acryo.ctf import CTFModel
from magicclass.logging import getLogger
from magicclass.utils import thread_worker

_Logger = getLogger("cylindra")


@thread_worker.callback
def on_ctf_finished(ctf_model: CTFModel | None = None, scale: float = 1.0):
    """Plot the simulated CTF image."""
    import matplotlib.pyplot as plt

    if ctf_model is None:
        return
    with _Logger.set_plt():
        ctf_image = ctf_model.simulate_image((128, 128), scale=scale)
        _Logger.print_html(
            f"CTF was simulated with parameters:<br>"
            f"Spherical aberration = {ctf_model.spherical_aberration:.2f} mm<br>"
            f"Defocus = {ctf_model.defocus:.2f} μm"
        )
        plt.figure()
        plt.imshow(np.fft.fftshift(ctf_image), cmap="gray")
        plt.title("Simulated CTF")
        plt.show()
