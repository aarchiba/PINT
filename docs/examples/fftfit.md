---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pint.profile.fftfit_aarchiba as fftfit
```

```python
template = np.zeros(256)
template[:16] = 1
plt.plot(np.linspace(0, 1, len(template), endpoint=False), template)
up_template = fftfit.upsample(template, 16)
plt.plot(
    np.linspace(0, 1, len(up_template), endpoint=False), up_template
)
plt.xlim(0, 1)
```

```python
template = np.diff(
    scipy.stats.vonmises(100).cdf(np.linspace(0, 2 * np.pi, 1024 + 1))
)
plt.plot(np.linspace(0, 1, len(template), endpoint=False), template)
up_template = fftfit.upsample(template, 16)
plt.plot(
    np.linspace(0, 1, len(up_template), endpoint=False), up_template
)
plt.plot(
    np.linspace(0, 1, len(template), endpoint=False),
    fftfit.shift(template, 0.25),
)
plt.xlim(0, 1)
```

```python
if False:
    template = np.diff(
        scipy.stats.vonmises(10).cdf(
            np.linspace(0, 2 * np.pi, 64 + 1)
        )
    )
    profile = fftfit.shift(template, 0.25)
else:
    template = np.random.randn(64)
    profile = np.random.randn(len(template))

upsample = 8
if len(template) != len(profile):
    raise ValueError(
        "Template is length %d but profile is length %d"
        % (len(template), len(profile))
    )
t_c = np.fft.rfft(template)
p_c = np.fft.rfft(profile)
ccf_c = np.zeros((len(template) * upsample) // 2 + 1, dtype=complex)
ccf_c[: len(t_c)] = t_c
ccf_c[: len(p_c)] *= np.conj(p_c)
ccf = np.fft.irfft(ccf_c)
x = np.argmax(ccf) / len(ccf)
l, r = x - 1 / len(ccf), x + 1 / len(ccf)

plt.figure()
xs = np.linspace(0, 1, len(ccf), endpoint=False)
plt.plot(xs, ccf)
plt.axvspan(r, l, alpha=0.2)
plt.axvline(x)


def gof(x):
    return (
        -(ccf_c * np.exp(2.0j * np.pi * np.arange(len(ccf_c)) * x))
        .sum()
        .real
    )


plt.plot(xs, [-2 * gof(x) / len(xs) for x in xs])

plt.figure()
xs = np.linspace(x - 4 / len(ccf), x + 4 / len(ccf), 100)
plt.plot(xs, [gof(x) for x in xs])
plt.axvspan(l, r, alpha=0.2)
plt.axvline(x)
```

```python
template = fftfit.upsample(
    np.diff(
        scipy.stats.vonmises(10).cdf(
            np.linspace(0, 2 * np.pi, 16 + 1)
        )
    ),
    2,
)
for s in np.linspace(0, 1 / len(template), 33):
    profile = fftfit.shift(template, s)
    print(
        (s - fftfit.fftfit_basic(template, profile)) * len(template)
    )
```

```python
a = np.random.randn(256)
a_c = np.fft.rfft(a)
a_c[-1] = 0
a_c[0] = 0
xs = np.linspace(0, 1, len(a), endpoint=False)
a_m = (
    (
        a_c[:, None]
        * np.exp(
            2.0j * np.pi * xs[None, :] * np.arange(len(a_c))[:, None]
        )
    )
    .sum(axis=0)
    .real
    * 2
    / len(a)
)
a_i = np.fft.irfft(a_c)
plt.plot(xs, a_m)
plt.plot(xs, a_i)
np.sqrt(np.mean((a_m - a_i) ** 2))
```

```python
c = np.zeros(6, dtype=complex)
c[-1] = 1
np.fft.irfft(c)
```

```python
r = np.random.randn(256)
r_c = np.fft.rfft(r)
r_1 = np.fft.irfft(np.conj(r_c))
plt.plot(r)
plt.plot(r_1[1::-1])
```

```python
n = 16
c = np.zeros(5, dtype=complex)
c[0] = 1
print(fftfit.irfft_value(c, 0, n))
np.fft.irfft(c, n)
```

```python
n = 8
c = np.zeros(5, dtype=complex)
c[-1] = 1
print(fftfit.irfft_value(c, 0, n))
np.fft.irfft(c, n)
```

```python
n = 16
c = np.zeros(5, dtype=complex)
c[-1] = 1
print(fftfit.irfft_value(c, 0, n))
np.fft.irfft(c, n)
```

```python
a = np.ones(8)
a[::2] *= -1
fftfit.shift(fftfit.shift(a, 1 / 16), -1 / 16)
```

```python
s = 1 / 3
t = fftfit.vonmises_profile(10, 16)
t_c = np.fft.rfft(t)
t_s_c = np.fft.rfft(fftfit.shift(t, s))
ccf_c = np.conj(t_c) * t_s_c
ccf_c[-1] = 0
plt.plot(np.fft.irfft(ccf_c, 256))
```

```python
s = 1 / 8
kappa = 1.0
n = 4096
template = fftfit.vonmises_profile(kappa, n)
profile = fftfit.shift(template, s / n)
rs = fftfit.fftfit_basic(template, profile)
print(s, rs * n)
upsample = 8

n_long = len(template) * upsample
t_c = np.fft.rfft(template)
p_c = np.fft.rfft(profile)
ccf_c = t_c.copy()
ccf_c *= np.conj(p_c)
ccf_c[0] = 0
ccf_c[-1] = 0
ccf = np.fft.irfft(ccf_c, n_long)
i = np.argmax(ccf)
assert ccf[i] >= ccf[(i - 1) % len(ccf)]
assert ccf[i] >= ccf[(i + 1) % len(ccf)]
x = i / len(ccf)
l, r = x - 1 / len(ccf), x + 1 / len(ccf)


def gof(x):
    return -fftfit.irfft_value(ccf_c, x, n_long)


print(l, gof(l))
print(x, gof(x))
print(r, gof(r))
print(-s / n, gof(-s / n))

res = scipy.optimize.minimize_scalar(
    gof, bracket=(l, x, r), method="brent", tol=1e-10
)
res
```

```python
t = fftfit.vonmises_profile(10, 1024, 1 / 3)
plt.plot(np.linspace(0, 1, len(t), endpoint=False), t)
plt.xlim(0, 1)
```

```python
profile1 = fftfit.vonmises_profile(1, 512, phase=0.3)
profile2 = fftfit.vonmises_profile(10, 1024, phase=0.7)
s = fftfit.fftfit_basic(profile1, profile2)
fftfit.fftfit_basic(fftfit.shift(profile1, s), profile2)
```

Okay, so let's try to work out the uncertainties on the outputs.

Let's view the problem as this: we have a set of Fourier coefficients $t_j$ for the template and a set of Fourier coefficients $p_j$ for the profile. We are looking for $a$ and $\phi$ that minimize

$$ \chi^2 =  \sum_{j=1}^m \left|ae^{2\pi i j \phi} t_j - p_j\right|^2. $$

Put another way we have a vector-valued function $F(a,\phi)$ and we are trying to match the observed profile vector. We can estimate the uncertainties using the Jacobian of $F$.

$$\frac{\partial F}{\partial a}_j = e^{2\pi i j \phi} t_j, $$

and

$$\frac{\partial F}{\partial \phi}_j = a 2\pi i j e^{2\pi i j \phi} t_j. $$

If this forms a matrix $J$, and the uncertainties on the input data are of size $\sigma$, then the covariance matrix for the fit parameters will be $\sigma^2(J^TJ)^{-1}$.


```python
n = 8

r = []
for i in range(10000):
    t = np.random.randn(n)
    t_c = np.fft.rfft(t)

    r.append(
        np.mean(np.abs(t_c[1:-1]) ** 2)
        / (n * np.mean(np.abs(t) ** 2))
    )
np.mean(r)
```

```python
template = fftfit.vonmises_profile(1, 256)
plt.plot(template)
plt.xlim(0,len(template))
std = 1
shift = 0
scale = 1
r = fftfit.fftfit_full(template, scale*fftfit.shift(template, shift), std=std)
r.shift, r.scale, r.offset, r.uncertainty, r.cov
```

```python
def gen_shift():
    return fftfit.wrap(
        fftfit.fftfit_basic(
            template, scale*template + std * np.random.randn(len(template))
        )
    )


shifts = []
```

```python
for i in range(1000):
    shifts.append(gen_shift())
np.std(shifts)
```

```python
r.uncertainty/np.std(shifts)
```

```python
scipy.stats.binom.isf(0.025, 100, 0.75), scipy.stats.binom.isf(0.975, 100, 0.75)
```

```python
scipy.stats.binom?
```
