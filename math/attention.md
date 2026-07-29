# Attention — Mathematical Note

Normative equations: [`spec/attention.md`](../spec/attention.md).

## Scaled dot-product

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}+M\right)V
\]

Scaling by \(\sqrt{d}\) keeps score variance ~1 so Softmax is neither flat nor
saturated (Vaswani et al.).

## Stable Softmax

\[
\mathrm{softmax}(z)_i=\frac{e^{z_i-\max_j z_j}}{\sum_k e^{z_k-\max_j z_j}}
\]

## GQA broadcast

For query head \(h\), KV head index is \(\lfloor h / (H/H_{kv}) \rfloor\).

## Complexity

Naive attention: time \(O(BHS^2d)\), score memory \(O(BHS^2)\).
