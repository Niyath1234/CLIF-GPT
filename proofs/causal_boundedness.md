# Causal Boundedness (CILF-ACI)

See [causal_boundedness.tex](causal_boundedness.tex) for the formal statement and proof sketch.

## Theorem (Conditional KL bound)

Let $P^*(w) \propto P_{\text{text}}(w)\exp(-E_{\text{cause}}(w,s))$ be the ideal causal next-token distribution,
and $P_F(w) \propto \exp(z_L(w) + \alpha z_J(w))$ the CILF-ACI fused distribution with
$z_J = W_U M a^*$ and $a^*$ minimizing the integrated physics energy $E(a)$.

Under:

1. $f_\psi$ is $L$-Lipschitz in $(s,a)$ over horizon $T$.
2. $\|M\|_2 \leq B_M$ and $\|W_U\|_2 \leq B_U$.
3. Adjoint inversion satisfies $\|\nabla_a E(a^*)\| \leq \epsilon_{\text{inv}}$.

Then:

$$\mathrm{KL}(P^* \| P_F) \leq \epsilon_{\text{inv}} B_U B_M + \lambda_{\text{cal}} + C(\alpha),$$

where $\lambda_{\text{cal}}$ is the projector calibration error from aligning $M\Delta s$ with token embeddings,
and $C(\alpha)$ vanishes as $\alpha \to 1$ when the gate saturates on high-energy scenes.

Numerical verification: `scripts/verify_causal_bound.py`.
