## Why L1 on Sigmoid Gates Encourages Sparsity

In this model, each weight is multiplied by a gate value produced by a sigmoid. That gate always stays between 0 and 1, so it can smoothly suppress or keep a connection instead of making a hard on/off decision.

The sparsity term uses the L1 norm of all gate values, which in this case is simply their sum since all values are non-negative.

The key behavior comes from how L1 works. It applies a constant shrink pressure regardless of a gate's current magnitude (its gradient is essentially constant, ±1). So a gate at 0.4 and a gate at 0.04 both receive the same push to decrease. In contrast, L2 regularization applies pressure proportional to magnitude, which tends to leave many small nonzero values instead of eliminating them.

Because of this, L1 is much more effective at producing true sparsity. Over training, gates that are not useful keep getting pushed down toward zero, while only the more important connections retain higher values.

## Results Table

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|------------------:|-------------------:|
| 1e-5   | 55.19             | 1.89               |
| 1e-3   | 51.20             | 95.12              |
| 1e-1   | 34.39             | 99.99              |

*Sparsity is measured as the percentage of gates with value below 1e-2.*

## Analysis

After running for 20 epochs, the expected trade-off becomes quite clear. With low sparsity pressure (1e-5), the model reaches the best accuracy (55.19%) but performs almost no pruning (1.89%). At a moderate setting (1e-3), accuracy drops slightly to 51.20%, while sparsity increases sharply to 95.12%, which gives a strong balance between compression and performance.

At a high setting (1e-1), nearly all connections are removed (99.99% sparsity), and accuracy drops significantly to 34.39%.

It is also worth noting that the baseline accuracy of around 55% is expected. CIFAR-10 is not particularly well-suited for a plain MLP without convolutional layers, so this limitation comes from the architecture rather than the pruning approach.

Looking at the gate distribution, there is a clear spike near zero, which confirms that strong pruning is happening. In my runs, this effect was quite noticeable in the histogram. The remaining active gates are spread roughly between 0.1 and 0.7, rather than clustering near 1.0.

This happens because the sigmoid function approaches 1.0 asymptotically but never actually reaches it. Under moderate regularization, the surviving gates settle at intermediate values instead of forming a sharp peak. Applying a hard threshold after training would likely make this separation cleaner.

Overall, this is exactly what we would expect from self-pruning gates. The regularization coefficient controls how aggressively the model removes connections. In this setup, 1e-3 appears to be the most practical choice, as it preserves reasonable accuracy while pruning about 95% of the network.

## Gate Distribution Plot

![Gate Distribution](gate_distribution.png)-