# Short-Answer Questions

## Part 1.1a
Let $c = \sqrt{a^2 + b^2}$, with $\phi=\tan^{-1}(a/b)$. See then
that $a = c sin(\phi)$ and $b = c cos(\phi)$. From the angle addition theorem:
$$
\begin{align*}
    y(x) &= c sin(\phi) cos(x) + c cos(\phi) sin(x) \\
         &= c sin(x + \phi)
\end{align*}
$$
This shows both the desired result and the amplitude ($c$) and phase shift ($\phi$)
of the resulting sinusoid.

## Part 1.1b
The norm of $y$ under this norm is:
$$
\begin{align*}
    \|y\| &= \frac{1}{\pi} \int_{-\pi}^{\pi} \sin^2(x + \phi)dx
          &= \frac{1}{\pi} \int_{-\pi}^{\pi} \sin^2(x)dx \\ 
          &= \frac{1}{\pi} \int_{-\pi}^{\pi} \frac{1}{2}(1 - \cos(2x))dx \\ 
          &= 1 - \frac{1}{\pi} \int_{-\pi}^{\pi} \frac{1}{2}\cos(2x)dx \\ 
          &= 1 - - \frac{1}{\pi} \sin(2x)/2|^{x=\pi}_{x=-\pi} \\
          &= 1
\end{align*}
$$
Where the second equality is from the periodicity of $\sin^2$ and the last
equality is from the 

## Part 2.3


## Part 2.5


## Part 3.4
