---
title: "PROTOTYPE: math transport"
---

# Representative notation

Energy-scan contraction follows $D(E)\propto 1/\sqrt{E}$.

The common Gaussian model is

$$
I(x,y)=B+A\exp\left[-\frac{1}{2}\left(
\frac{u^2}{\sigma_x^2}+\frac{v^2}{\sigma_y^2}
\right)\right].
$$

Its center $(x_c,y_c)$, widths $(\sigma_x,\sigma_y)$, and angle
$\theta$ remain explicit.

The pair midpoint is $m_{ij}(E)=[p_i(E)+p_j(E)]/2$.

LocoTrack availability uses

$$
P_{\mathrm{vis}}=[1-\operatorname{sigmoid}(o)]
[1-\operatorname{sigmoid}(d)],
$$

with $P_{\mathrm{vis}}>0.5$.

Lucas--Kanade accepts $r\leq30$ and $e_{\mathrm{FB}}\leq8\,\mathrm{px}$,
after high-pass filtering with $\sigma=5$.

EasyLEED uses $R^2\geq0.8$ and $\gamma=8$.

