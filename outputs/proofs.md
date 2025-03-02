# Short-Answer Questions

## Part 1.1a
Let $c = \sqrt{a^2 + b^2}$, with $\phi=\tan^{-1}(a/b)$. See then
that $a = c \sin(\phi)$ and $b = c \cos(\phi)$. From the angle addition theorem:

$$
\begin{align*}
    y(x) &= c \sin(\phi) \cos(x) + c \cos(\phi) \sin(x) \\
         &= c \sin(x + \phi)
\end{align*}
$$

This shows both the desired result and the amplitude ($c$) and phase shift ($\phi$)
of the resulting sinusoid.

## Part 1.1b
If $c$ is on the unit circle, the norm of $y$ under this norm is:

$$
\begin{align*}
    \|y\| &= \frac{1}{\pi} \int_{-\pi}^{\pi} \sin^2(x + \phi)dx \\
          &= \frac{1}{\pi} \int_{-\pi}^{\pi} \sin^2(x)dx \\ 
          &= \frac{1}{\pi} \int_{-\pi}^{\pi} \frac{1}{2}(1 - \cos(2x))dx \\ 
          &= 1 - \frac{1}{\pi} \int_{-\pi}^{\pi} \frac{1}{2}\cos(2x)dx \\ 
          &= 1 - \frac{1}{4\pi} \sin(2x)|^{x=\pi}_{x=-\pi} \\
          &= 1
\end{align*}
$$

Where the second equality is from the periodicity of $\sin^2$.

## Part 2.3
These eigenvectors are more irregular than those of the Toeplitz matrix from
lecture as they correspond to 2d fourier basis vectors plotted in a 1d manner
(with the rows flattened and concatenated). If we were to plot these in two
dimensions then the corresponding image would look much more regular. In descending
order, the eigenvalues correspond to eigenvectors that in turn correspond to
eigenfunctions with higher frequencies.

## Part 2.5
The operators are block diagonalized. This is expected because circulant
matrices, such as the shift operator, are block diagonalizable in the 2D fourier
domain. This is desirable because it is easier to compute products with the
matrix, meaning that it is efficient (and straightforward) to perform these
kinds of operations in the fourier domain.

## Part 3.4
The outputs are the same because the convolution operator is linear. Convolving
the image with a linear combination of kernels is the same as linearly combining
the convolution of the image with each kernel. This tells us that the same
rotation in image space vs the space of oriented filters yields the same final
result. The fact that these are identical tells us that rotating in image space
corresponds to rotating in the fourier basis (i.e., in frequency space).


