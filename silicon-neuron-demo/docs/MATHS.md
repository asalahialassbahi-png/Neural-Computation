# The mathematics of the Silicon Neuron project

Everything the code does, derived from first principles: the artificial network,
the spiking neuron, how to train something that is not differentiable, how the
Pico-optimised spiking network (SNN-E) decides to stop early, and the energy
inequality that decides which network wins on a Raspberry Pi Pico. Each section
names the file that implements it, so every equation can be checked against
running code.

Notation: vectors are bold, $n$ counts time steps, $i$ input pixels, $j$ hidden
neurons, $k$ output neurons (digits). $B$ is a mini-batch size.

---

## 1. The task

MNIST: 28 × 28 greyscale digits, pixel values $x_i \in \{0,\dots,255\}$, so
$N=784$ inputs, 10 classes. 60,000 training images (we hold out the last 5,000
for validation and never train on them) and 10,000 test images that are used
**once**, at the end. Every network in the project has the same shape,
784 → 300 → 10, so the comparison is between ways of computing, not between
sizes.

---

## 2. The artificial neural network (ANN) — `training/models.py: ANN`

### 2.1 Forward pass

$$\mathbf{z}^{(1)} = W^{(1)\top}\mathbf{x} + \mathbf{b}^{(1)},\qquad
\mathbf{h} = \mathrm{ReLU}(\mathbf{z}^{(1)}) = \max(0,\mathbf{z}^{(1)}),\qquad
\mathbf{z}^{(2)} = W^{(2)\top}\mathbf{h} + \mathbf{b}^{(2)}$$

with $\mathbf{x}$ scaled to $[0,1]$. Each hidden neuron does 784 multiply-adds
and each output 300, so one digit costs

$$N_{\mathrm{MAC}} = 784\times 300 + 300 \times 10 = 238{,}200 .$$

### 2.2 Softmax and cross-entropy

$$p_k = \frac{e^{z_k}}{\sum_m e^{z_m}},\qquad L = -\log p_y = -z_y + \log\sum_m e^{z_m}.$$

The code subtracts $\max_m z_m$ before exponentiating (it cancels in the ratio)
so that $e^{z}$ never overflows.

### 2.3 The gradient (partial differentiation)

$$\frac{\partial L}{\partial z_k} = -\delta_{ky} + \frac{e^{z_k}}{\sum_m e^{z_m}} = p_k - \delta_{ky},$$

the famous "prediction minus target". The chain rule takes it backwards:

$$\frac{\partial L}{\partial W^{(2)}_{jk}} = h_j\,(p_k-\delta_{ky}),\qquad
\frac{\partial L}{\partial h_j} = \sum_k W^{(2)}_{jk}(p_k-\delta_{ky}),\qquad
\frac{\partial L}{\partial z^{(1)}_j} = \frac{\partial L}{\partial h_j}\,\mathbb{1}[z^{(1)}_j>0],$$

because $\mathrm{ReLU}'(z) = H(z)$, the Heaviside step. In matrix form over a
batch this is two matrix products per layer, which is exactly `ANN.backward`.

### 2.4 Optimiser (Adam)

With gradient $g_t$ at step $t$:
$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$,
$\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$,
$W \leftarrow W - \eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$
(β₁ = 0.9, β₂ = 0.999). `models.py: Adam`.

---

## 3. The spiking neuron — `training/models.py: SNN`

### 3.1 From a circuit to a difference equation (discretisation)

A leaky integrate-and-fire (LIF) neuron is an RC circuit: membrane voltage $U$
leaks through a resistor while input current charges the capacitor,

$$\tau\,\frac{dU}{dt} = -U(t) + R\,I(t).$$

Over one time step $\Delta t$, holding $I$ constant, the exact solution is

$$U(t+\Delta t) = e^{-\Delta t/\tau}\,U(t) + \left(1-e^{-\Delta t/\tau}\right) R I .$$

Writing $\beta = e^{-\Delta t/\tau}$ and absorbing the constant $(1-\beta)R$
into the learned weights gives the discrete neuron used everywhere:

$$U[n] = \beta\,U[n-1] + I[n].$$

The neuron **spikes** when $U$ crosses a threshold $\theta$,
$S[n] = H(U[n]-\theta)$, and is then **reset by subtraction**, which keeps
the overshoot:

$$\boxed{U[n] = \beta\,U[n-1] + I[n] - \theta\,S[n-1]}$$

### 3.2 Two neuron models

* **Second-order (current-based) LIF**, used by SNN-L and SNN-P: the input
  current is itself a leaky trace, $I[n] = \alpha I[n-1] + \sum_i W_{ij}s_i[n]$,
  with $\alpha = e^{-\Delta t/\tau_{\mathrm{syn}}}$. Two state variables per neuron.
* **First-order LIF**, used by SNN-E: $\alpha=0$, so $I[n]=\sum_i W_{ij}s_i[n]$.
  One state variable per neuron, and the input can be added **straight into
  the membrane**. Section 8 shows why this matters on a Pico.

### 3.3 Leaks as bit shifts

The Pico has no floating point. Choosing $\beta = 1-2^{-K}$ turns the leak into
one shift and one subtraction:

$$\beta U = U - 2^{-K}U \;\Rightarrow\; U \leftarrow U - (U \gg K).$$

With $K=3$, $\beta = 0.875$, i.e. $\tau = -\Delta t/\ln\beta \approx 7.5$ steps.
(`config.py: K_BETA`, `firmware/nn_core.c`.)

### 3.4 The output layer

Ten non-spiking integrators read the hidden spikes:
$U_2[n] = \beta U_2[n-1] + \sum_j V_{jk} S_j[n]$, and the evidence for digit $k$
is accumulated,

$$c_k[n] = \sum_{m\le n} U_{2,k}[m],\qquad \hat y = \arg\max_k c_k[T-1].$$

---

## 4. Turning pixels into spikes — `training/encoding.py`

### 4.1 Latency (time-to-first-spike) code

A brighter pixel spikes **earlier**, exactly **once**:

$$t_i = \left\lfloor \frac{(255-x_i)\,T_{\mathrm{in}}}{256}\right\rfloor \quad\text{if } x_i \ge x_{\min},\quad\text{no spike otherwise.}$$

The number of input spikes is the number of pixels above $x_{\min}$: never more
than the ANN's non-zero pixels. SNN-E uses $T_{\mathrm{in}}=3$, $x_{\min}=96$.

### 4.2 Rate (Poisson) code

Each step, pixel $i$ spikes with probability $x_i/256$. Over $T$ steps the
expected number of input spikes is

$$\mathbb{E}[N_{\mathrm{spk}}] = T\sum_i \frac{x_i}{256} \approx 16 \times 104 \approx 1{,}700,$$

**15 times more** events than the latency code for the same digit. Section 8
turns that directly into energy, and it is why SNN-P is the slowest network.

---

## 5. Training a spiking network — `models.py: SNN.backward`

### 5.1 The problem

$S = H(U-\theta)$ has derivative $\delta(U-\theta)$: zero almost everywhere,
infinite at the threshold. Gradient descent gets no signal.

### 5.2 The surrogate gradient

Keep the exact step in the forward pass, but in the backward pass replace its
derivative by a smooth bump (SuperSpike, Zenke & Ganguli 2018):

$$\frac{\partial S}{\partial U} \approx \sigma'(U-\theta) = \frac{1}{\left(k\,|U-\theta|+1\right)^2},\qquad k=10.$$

### 5.3 Back-propagation through time (BPTT)

Unroll the network over $T$ steps; it becomes a deep network with shared
weights. Define, for one hidden neuron,
$\varepsilon[n] = \partial L/\partial U[n]$ and $\psi[n] = \partial L/\partial I[n]$,
and let $g[n] = \partial L/\partial S[n]$ be what arrives from the layer above.
$U[n]$ affects the loss through its own spike $S[n]$, through
$U[n+1] = \beta U[n] + \dots$, and through the reset $-\theta S[n]$ in
$U[n+1]$. The chain rule gives

$$\varepsilon[n] = \sigma'(U[n]-\theta)\,\big(g[n] - \theta\,\varepsilon[n+1]\big) + \beta\,\varepsilon[n+1],$$

and because $I[n]$ feeds $U[n]$ and $I[n+1]=\alpha I[n]+\dots$,

$$\psi[n] = \varepsilon[n] + \alpha\,\psi[n+1],\qquad \varepsilon[T]=\psi[T]=0.$$

Finally $I[n] = \alpha I[n-1] + \sum_i W_{ij}s_i[n]$ gives the weight gradient and
the signal passed to the previous layer:

$$\frac{\partial L}{\partial W_{ij}} = \sum_n s_i[n]\,\psi_j[n],\qquad
\frac{\partial L}{\partial s_i[n]} = \sum_j W_{ij}\,\psi_j[n].$$

(For SNN-E, $\alpha=0$ and $\psi=\varepsilon$.)

### 5.4 Sparsity regulariser

The loss adds $\lambda\,\overline{S}$, the mean firing rate, whose gradient is
the constant $\lambda/(BTH)$ added to every $g[n]$. Fewer spikes means less work
downstream (section 8).

### 5.5 Proof that the derivation is right — `training/gradcheck.py`

Replace the step by a steep sigmoid in the forward pass so that the network is
differentiable, and use its exact derivative in the backward pass. Then compare
the analytic directional derivative $\langle \nabla L, \mathbf d\rangle$ with a
central difference $\big(L(W+\epsilon\mathbf d)-L(W-\epsilon\mathbf d)\big)/2\epsilon$.
The truncation error of a central difference is $O(\epsilon^2)$, so the error
must fall about 100 times for every 10 times smaller $\epsilon$, until
floating-point round-off ($\sim 10^{-16}L/\epsilon$) takes over. Measured:

| network | best relative error |
|---|---|
| ANN | 2.5 × 10⁻⁸ |
| SNN (second order, mean readout) | 3.6 × 10⁻⁹ |
| SNN-E (first order, per-step readout) | 4.4 × 10⁻⁸ |

A wrong term anywhere in the recursions would leave an error near 10⁻² that
does not move with $\epsilon$.

---

## 6. SNN-E: answering early — `models.py` (readout = "perstep"), `intsim.py: snn_fast_int`

### 6.1 A loss at every time step

To be allowed to stop at step $n$, the network must already be right at step
$n$. So instead of only scoring the final evidence, SNN-E is trained on the
average of the losses at every step, each on the mean evidence so far:

$$z_k^{(n)} = \frac{c_k[n]}{n+1},\qquad L = \frac{1}{T}\sum_{n=0}^{T-1} \mathrm{CE}\!\left(\mathbf z^{(n)}, y\right).$$

### 6.2 Its gradient

From section 2.3, $\partial L/\partial z_k^{(n)} = (p_k^{(n)}-\delta_{ky})/T$, so
$\partial L/\partial c_k[n] = (p_k^{(n)}-\delta_{ky})/\big(T(n+1)\big)$. Since
$c[n] = \sum_{m\le n} U_2[m]$, the output potential at step $m$ appears in
every $c[n]$ with $n\ge m$:

$$\frac{\partial L}{\partial U_{2}[m]} = \sum_{n\ge m} \frac{\partial L}{\partial c[n]},$$

a reverse cumulative sum. That is the only change to the BPTT of section 5.3,
and gradcheck.py verifies it (table above).

### 6.3 The early-exit rule

After each step, let $c_{(1)}$ and $c_{(2)}$ be the largest and second-largest
evidence. Stop when the leader is far enough ahead **on average per step**:

$$c_{(1)}[n] - c_{(2)}[n] \;\ge\; m\,(n+1).$$

Comparing with $m(n+1)$ rather than $m$ keeps the rule in the units of the
trained logits $z^{(n)}$. The margin $m$ is a whole number of thresholds
(SNN-E: $m = 3\theta$), chosen on the **validation** images as the cheapest
margin for which stopping early **changes the answer on at most 0.1% of the
images** compared with always running all 8 steps. (Comparing two accuracies
measured on 5,000 images would be noisier: sampling alone moves an accuracy by
about $\pm\sqrt{p(1-p)/5000}\approx\pm0.2$ points, while agreement between two
predictions of the same images is measured almost exactly.) On the test set the
network then stops after **1.38 of 8 steps on average**.

---

## 7. Integers on the Pico — `training/intsim.py`, `firmware/nn_core.c`

* Weights: symmetric int8, $W_q = \mathrm{round}(W/s)$, $s=\max|W|/127$.
* SNN state is int32 in units of $s$, so $\theta_q = \mathrm{round}(\theta/s)$
  (SNN-E: $\theta_q = 86$), and every leak is a shift (section 3.3).
* ANN hidden activations are re-quantised to uint8 with one integer multiply
  and shift: $h_8 = \mathrm{clamp}\big((a\,M + 2^{23}) \gg 24,\,0,255\big)$.
* `verify_c.py` compiles the firmware's own C on a PC and checks that every
  prediction, spike count, step count and output potential is identical to the
  Python integer model, for all 500 digits stored on the Pico: **0 differences**.

---

## 8. Energy: why a network wins or loses on a Pico

### 8.1 The textbook argument (custom silicon)

On 45 nm CMOS (Horowitz 2014) an 8-bit add costs 0.03 pJ and an 8-bit multiply
0.2 pJ, so a multiply-accumulate costs about 7 times an accumulate:

$$E_{45} = N_{\mathrm{MAC}}(E_\times + E_+) + N_{\mathrm{AC}}\,E_+ .$$

By this measure every SNN, even the 16-step one, looks many times cheaper than
the ANN. That is the argument in most SNN papers, and it is true for
neuromorphic chips.

### 8.2 The Pico is different

The RP2040 has a single-cycle hardware multiplier. Compiling the firmware's
inner loops (`arm-none-eabi-gcc -O2 -mcpu=cortex-m0plus`) and counting the
Cortex-M0+ cycles:

| inner loop | instructions | cycles |
|---|---|---|
| ANN multiply-accumulate `acc += x * w` | ldrsb, ldr, muls, adds, stmia, adds, cmp, bne | 13 |
| SNN accumulate `U += w` | ldr, ldrsb, adds, adds, stmia, cmp, bne | 12 |
| SNN-L neuron update (current leak + membrane) | — | 40 per neuron per step |
| SNN-E fused threshold / leak / reset | — | 17 per neuron per step |

**An addition saves 1 cycle in 13.** On the Pico, a spiking network can only
win by doing *fewer* operations, not cheaper ones.

### 8.3 Energy from cycles

To first order the Pico's energy per digit is

$$E \approx P_{\mathrm{active}}\cdot\frac{\text{cycles}}{f_{\mathrm{clk}}},$$

so cycles are a proxy for energy. The real value is measured by the INA226
(`pi/bench.py`); it also includes flash reads, which scale with the same row
counts.

### 8.4 The break-even inequality

Per digit, with $N_{\neq 0}$ non-zero pixels (the zero-skipping ANN's work),
$e$ input spikes, $s$ time steps actually run and $H = 300$:

$$\underbrace{e\,(r + H c_{+})}_{\text{input spikes}} + \underbrace{s\,H\,c_{n}}_{\text{neuron upkeep}}
\;<\; \underbrace{N_{\neq0}\,(r' + H c_{\times})}_{\text{ANN-Z layer 1}}$$

with $c_+ = 12$, $c_\times = 13$, $r, r' \approx 15$ cycles of row set-up and
$c_n$ the per-neuron, per-step upkeep. This one line explains every result:

* **SNN-L** has $e \approx 115 < N_{\neq0} \approx 141$, but $s = 16$ and
  $c_n = 40$ add $16\times300\times40 = 192{,}000$ cycles of upkeep:
  it **loses** narrowly to ANN-Z (677k vs 640k cycles, emulated).
* **SNN-P** has $e \approx 1{,}500$: it loses to everything.
* **SNN-E** attacks every term: one state variable ($c_n=17$), fewer input
  spikes ($x_{\min}=96$), and early exit ($s\approx1.4$). Every one of the
  10,000 test digits lies below the break-even line (`results/fig_break_even.png`).

### 8.5 Measured on the emulated Pico (`training/cycle_audit.py`)

The compiled firmware runs in an ARM emulator (Unicorn) on real test digits.
Each executed basic block is priced with the Cortex-M0+ timings, and the
emulated answers are checked against the Python model (0 mismatches):

| network | cycles per digit | vs dense ANN | vs ANN-Z |
|---|---|---|---|
| ANN (dense) | 3,367k | — | 5.26× more work |
| ANN-Z (skips zero pixels) | 640k | 5.26× less | — |
| SNN-L (latency, 16 steps) | 677k | 4.97× less | 1.06× **more** |
| SNN-P (rate code) | 6,241k | 1.85× **more** | 9.75× more |
| **SNN-E (Pico-optimised)** | **345k** | **9.77× less** | **1.86× less** |

The hand model of 8.4 predicts the emulator to within 4–9%.

---

## 9. Is the difference real? (statistics) — `training/prove.py`, `training/repro.py`

* **McNemar's test**: both networks classify the same 10,000 digits, so only the
  digits where exactly one is right carry information. Under "equally accurate",
  the number won by SNN-E out of those discordant digits is Binomial(n, ½).
* **Equivalence (TOST)**: "not significantly different" is not the same as "the
  same". TOST tests two one-sided hypotheses: the difference is above $-\Delta$
  and below $+\Delta$. If both are rejected, the networks are equivalent within
  $\pm\Delta$.
* **Bootstrap**: the confidence interval of the cycle ratio comes from
  resampling digits 10,000 times.
* **Repeatability**: `repro.py` retrains every network from scratch with five
  seeds and reports mean ± sd with a t-based 95% interval, plus a manifest of
  library versions and SHA-256 hashes of the data and code.

Results: `results/PROOF.md` and `results/repro.md`.

---

## 10. What the project shows

1. A spiking network **can** beat a conventional network on energy on an
   ordinary microcontroller, by **9.8×** against the standard dense ANN and
   **1.86×** against an ANN that skips zeros, while staying within half a
   percentage point of its accuracy.
2. It does **not** happen automatically. The textbook 16-step SNN loses to a
   sensible ANN on a Pico, and a rate-coded SNN loses to everything. The
   hardware decides which equations matter: on a chip with a fast multiplier,
   the SNN must win by doing fewer operations (latency code, one state variable,
   early exit), not cheaper ones.
3. The claims are checked three independent ways: the derivatives numerically
   (gradcheck), the firmware against the maths bit for bit (verify_c), and the
   energy on emulated and then real hardware (cycle_audit, INA226).

## References

* J. K. Eshraghian, M. Ward, E. O. Neftci, X. Wang, G. Lenz, G. Dwivedi,
  M. Bennamoun, D. S. Jeong, W. D. Lu, "Training Spiking Neural Networks Using
  Lessons From Deep Learning", *Proceedings of the IEEE* 111(9), 2023
  (the snnTorch paper: LIF discretisation, surrogate gradients, BPTT, readouts).
* F. Zenke, S. Ganguli, "SuperSpike: Supervised learning in multilayer spiking
  neural networks", *Neural Computation* 30(6), 2018 (the surrogate gradient).
* M. Horowitz, "Computing's energy problem (and what we can do about it)",
  ISSCC 2014 (the 45 nm operation energies).
* D. P. Kingma, J. Ba, "Adam: A method for stochastic optimization", ICLR 2015.
* Raspberry Pi Ltd, *RP2040 Datasheet* (single-cycle multiplier, XIP flash) and
  ARM, *Cortex-M0+ Technical Reference Manual* (instruction timings).
