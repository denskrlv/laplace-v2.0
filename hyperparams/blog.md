## Hyperparams sensitivity
...

We've designed a comprehensive sensitivity analysis to understand how different hyperparameter choices impact the performance of the Laplace Approximation (LA), specifically its ability to produce reliable uncertainty estimates. Our experiment is built around the **laplace** library introduced in the paper  and uses the paper's own findings to guide our choices, especially in the context of limited computational resources.

#### Architectures: WideLeNet and ResNet18

To determine whether our sensitivity findings generalize across network designs we apply the LA to two different convolutional architectures:

- **WideLeNet** is a modern yet shallow architecture. As a "widened" variant of the classic LeNet-5, it has strong locality priors from its convolutional filters but increases representational capacity by expanding its channel count. This design typically results in a particular kind of loss-surface geometry, characteristic of less deep networks.

- **ResNet18**, in contrast, is a substantially deeper architecture that has become a standard in computer vision. Its use of residual (or "skip") connections fundamentally alters the training dynamics and produces a more complex loss landscape, which is representative of many modern, high-performing models.

Using two different architectures allows us to check if our findings hold in general or are merely an artifact of a single model's structure. For our experiment, we trained both models on the MNIST dataset. This is a critical first step, as the post-hoc LA (which we are going to investigate) is designed to be applied to a pretrained model. We first find a Maximum a Posteriori (MAP) estimate through standard training, and then the LA builds a Gaussian approximation to the posterior around that point.

#### Data: In-Distribution vs. Out-Of-Distribution

A model's uncertainty is most tested when it encounters data it wasn't trained on. Therefore, evaluating on both in-distribution (ID) and out-of-distribution (OOD) data is essential.

- In-Distribution (**ID**): **MNIST** Test Set.
Our ID data is the standard MNIST test set. This gives us a baseline performance measure in the ideal scenario where the test data comes from the same distribution as the training data.

- Out-of-Distribution (**OOD**): **Rotated-MNIST** (R-MNIST).
For our OOD challenge, we chose Rotated-MNIST. This is a perfect choice for a sensitivity analysis and one used directly in the "Laplace Redux" paper for evaluating calibration under dataset shift (see Figure 4). Rather than using a completely different dataset, R-MNIST introduces a controlled and continuous "shift intensity". Our script evaluates the models on images rotated by a range of angles: 5, 15, 30, 45, 60, 90, 120, 160, and 180 degrees. This allows us to observe not just if performance degrades on OOD data, but how gracefully it does so as the data shifts further and further from the original distribution.

#### Metrics and experimental design

Simple accuracy is not enough to evaluate a Bayesian model, as a model can be accurate but dangerously overconfident. Our experiment focuses on metrics that directly measure the quality of uncertainty estimates, such as Negative Log-Likelihood (**NLL**) and Expected Calibration Error (**ECE**). NLL penalizes a model for being both incorrect and confident , while ECE directly measures if a model's confidence is reliable.

The central goal is to test sensitivity to hyperparameters. Given our limited resources, we made a strategic decision to focus on the last-layer LA by setting `subset_of_weights='last_layer'`. The paper repeatedly highlights the last-layer LA as a powerful and efficient variant. It is described as "cost-effective yet compelling" , significantly cheaper than applying the LA to all weights, and is the recommended default in the laplace library. By only treating the final layer's weights probabilistically, we dramatically reduce the size of the Hessian matrix that needs to be computed and inverted, making a wide hyperparameter sweep computationally feasible.

With that fixed, we created a grid to explore the most influential remaining hyperparameters:

- **`prior_precision`**:
This is a fundamental Bayesian hyperparameter, corresponding to the inverse variance of the prior distribution over the weights. A low precision implies a broad prior (less regularization), while high precision implies a narrow, restrictive prior. We are testing a wide logarithmic scale of values (1e-6, 1e-4, 1e-2, 1, 10, 100)  to see how strongly this choice affects the final calibration and OOD performance.

- **`hessian_structure`**:
We evaluate two options: `diag` and `kron`. The `diag` (diagonal) approximation is the most lightweight, assuming independence between all weights. In contrast, `kron` (Kronecker-factored / KFAC) is more expressive, modeling correlations between weights within the same layer, and is noted to provide a good trade-off between expressiveness and speed. Our experiment will directly reveal if the added complexity of KFAC provides a tangible benefit over the simpler diagonal approximation in the last-layer context.

- **`link_approx`**:
For classification, the model first produces intermediate scores called "logits." To turn these logits into final probabilities, a conversion step is needed. The exact math for this step is complex, so we use an approximation. This parameter lets us choose which one: 
The `probit` approximation replaces the logistic-sigmoid function with the probit function, which has a similar shape but renders the integral analytically solvable.
The `bridge` approximation provides a mapping from the Gaussian distribution over the logits to a Dirichlet distribution over the class probabilities.
Our experiment will serve to validate which of these approximations is more effective in our setup.

- **`temperature`**:
We also include temperature scaling, a common technique for post-hoc calibration. Its inclusion allows us to see how this simple method interacts with the more complex machinery of the LA. We test 3 different temperature values: 0.1, 0.5 and 1

To ensure our results are reliable, we've used multiple random seeds ([0, 42, 123]) for training our base models. Neural network training is a stochastic process, and a single run can be an outlier. By running our entire experiment across models trained with different seeds and then averaging the results, we ensure our conclusions are robust and not due to a fluke of random initialization.

### Prior Precision analysis

In our exploration, we can first zoom in on the most crucial hyperparameter: `prior_precision`. This single value, which controls the strength of our Bayesian regularization, has a profound impact on model calibration and robustness.

> Note: when the plots are titled Aggregated Result, results where averaged between the two different models (WideLeNet and ResNet18), as their behaviour was almost identical, thus not interesting to show both.

1.  **Regularization improves ID calibration**

Our first key finding relates to the solid black line present in all the plots (here we show just the most insightful ones to ensure clarity), representing performance on ID data. In every configuration, as we increase `prior_precision` from very small values, the performance on ID data either stays excellent or gets significantly better.

![ECE-kron-probit-pp](\blog_plots\precision\aggregated_ece_vs_prior_precision_context_hessiankron_linkprobit_avg_temp.png)

This is most evident in this Aggregated ECE plot for the `kron`/`probit` context. The ID ECE starts very high (indicating poor calibration) and then plummets to near-zero for `prior_precision` values of 10.0 or higher. This shows that a higher `prior_precision` acts as a powerful regularizer, penalizing overly complex solutions and forcing the model into a state that is much better calibrated on data it expects to see.

2.  **Critical trade-off for OOD robustness**

While high `prior_precision` is good for ID data, it comes at a cost. This brings us to the most important insight from our entire experiment, which is best illustrated by the Aggregated NLL plot for the `kron`/`probit` context.

![NLL-kron-probit-pp](\blog_plots\precision\aggregated_nll_vs_prior_precision_context_hessiankron_linkprobit_avg_temp.png)

This plot reveals a critical trade-off between performance on in-distribution data and robustness to out-of-distribution data, which is governed by the prior_precision.

For ID and low-shift data (up to ~55° rotation), performance consistently improves as `prior_precision` increases. The NLL for these curves continually decreases, reaching its minimum at the highest levels of regularization. This shows that for data the model is familiar with, strong regularization helps the model make more confident and accurate predictions.

For high-shift data, he behavior is completely different. The NLL for these curves follows a distinct U-shape. It is minimized at a relatively low `prior_precision` (around 1e-2 to 1.0) and then skyrockets as the precision increases further.

This creates a fundamental conflict. The strong regularization that is optimal for ID data makes the model "rigid" and over-specialized to the training distribution. When this rigid model is faced with heavily shifted data, it not only makes incorrect predictions but does so with high confidence, leading to a catastrophic NLL. Therefore, choosing a `prior_precision` is not about maximizing in-distribution performance, but about finding a balance that prevents such overconfident failures on unseen data, accepting a slight compromise on ID performance for the sake of OOD robustness.

3. **The full picture depends on the configuration**

Is this U-shaped trade-off universal? Not necessarily. The final insight is that the behavior of `prior_precision` is deeply connected to the other hyperparameters, namely `hessian_structure` and `link_approx`.

To see this, we should compare two ECE plots side-by-side.

| ECE vs. Prior Precision (kron/probit) | ECE vs. Prior Precision (diag/bridge) |
|:--:|:--:|
| ![kron/probit](\blog_plots\precision\aggregated_ece_vs_prior_precision_context_hessiankron_linkprobit_avg_temp.png) | ![diag/bridge](\blog_plots\precision\aggregated_ece_vs_prior_precision_context_hessiandiag_linkbridge_avg_temp.png) |

On the left (`kron`/`probit`), we see the dramatic U-shaped behavior for higly shifted OOD data. On the right (`diag`/`bridge`), the pattern is different. While the metrics still improve with higher precision, the OOD error for high-precision values tends to flatten out rather than sharply increasing.

This tells us that the combination of a more expressive Hessian (`kron`) and a `probit` link function creates a model that is highly sensitive and dynamic in its response to regularization. In contrast, the simpler `diag` Hessian and `bridge` link function lead to a more stable, perhaps less optimal, but more robust behavior against extreme regularization.

#### The curious case of the insensitive hyperparameter

In our experiment, one of the most valuable things we can uncover is not just how much a hyperparameter matters, but when it matters. In our exploratory plots, we noticed an interesting edge case: for the specific configuration using a Kronecker-factored Hessian (`hessian_structure='kron'`) and the Laplace Bridge predictive approximation (`link_approx='bridge'`), the choice of `prior_precision` appears to have virtually no effect on the model's final performance, be it NLL or ECE.

![flat-pp](\blog_plots\precision\special_case_summary_kron_bridge_horizontal_FINAL.png)

This phenomenon occurs at the final prediction step. The `prior_precision` is a critical parameter that directly shapes the variance of the Gaussian posterior over the model's weights. However, the influence of this variance on the final output is mediated by the `link_approx` and `hessian_structure`.

Our results suggest that the `bridge` approximation's sensitivity to the posterior variance is highly dependent on the structure of the covariance matrix it receives. When paired with the correlated, block-diagonal covariance from the `kron` Hessian, the `bridge`'s calculations appear to be overwhelmingly dominated by the posterior mean—the deterministic MAP prediction. In this specific combination, the nuanced variance information, which is controlled by `prior_precision`, is effectively disregarded.

This is a perfect example of why sensitivity analysis is so critical. It reveals that certain combinations of methods can lead to unexpected behaviors, where the effect of one hyperparameter is effectively nullified by another.

#### When models diverge

While we've seen several cases where our ResNet18 and WideLeNet models exhibit similar patterns, allowing us to summarize their behavior with an aggregated plot (like the ones showed before), this is not always the case. 

A prime example of this divergence is seen when using the (`hessian=diag`, `link=probit`) configuration.

![comparison-pp](\blog_plots\precision\model_comparison_ece_vs_prior_precision_context_hessiandiag_linkprobit_avg_temp.png)

In the plot above, which shows the ECE, the two models tell different stories:
- For ResNet18 , the ID ECE starts very high, around 0.4, indicating that the pre-trained model, before the full effect of the LA's regularization is applied, is poorly calibrated. As `prior_precision` increases, the ECE drops dramatically. Here, the Laplace Approximation has a powerful corrective effect, fixing a deficient baseline model.
- For WideLeNet, the ID ECE starts near-zero, indicating the pre-trained model is already very well-calibrated. As `prior_precision` changes, the LA's main role is to maintain this excellent calibration. The changes are far less dramatic because there was no fundamental problem to solve.

This divergence isn't a failure of the method but a reflection of the models themselves. The Laplace Approximation is a post-hoc method applied to an existing, pre-trained model. The characteristics of that baseline model matter: 
- ResNet18, a deep and highly complex architecture, may be prone to overfitting or learning less robust features on a comparatively simple dataset like MNIST, resulting in poor initial calibration. 
- WideLeNet, being a wider but shallower architecture, may have hit a sweet spot of capacity for this task, leading to a naturally well-calibrated solution.

Hence, LA is not a one-size-fits-all tool, its effect is context-dependent and heavily influenced by the quality of the initial MAP estimate. By keeping the results for this configuration separate, we highlight the LA's versatility, it can both fix poor models and preserve the quality of good ones.

### Hessian Structure and Link Approximation analysis

The "Laplace Redux" paper states that the Kronecker-factored (`kron`) Hessian approximation offers a good trade-off between expressiveness and efficiency compared to a simple diagonal (`diag`) one. Our analysis reveals this is not a universal truth, but rather a complex reality where the optimal Hessian structure depends on a three-way interplay between regularization (`prior_precision`), the chosen link approximation (`link_approx`), and the severity of the distribution shift.

#### The unifying effect of strong regularization

Across all configurations, one finding is absolute: at high `prior_precision` (e.g., 100.0), the choice of Hessian structure becomes mostly irrelevant. As seen in the right-hand plots of the next plots, the performance lines for `diag` and `kron` almost converge. This is because a strong prior dominates the posterior estimation. The resulting Gaussian approximation of the posterior is so constrained by the narrow prior that the subtle details of the loss landscape's curvature, which the Hessian is meant to capture, become negligible. In this high-regularization regime, the simpler and more efficient `diag` approximation is sufficient. The truly interesting and complex behavior unfolds at low `prior_precision`, where the data likelihood has a much stronger influence on the posterior shape.

#### The role of the Link Approximation at low regularization

At low `prior_precision` (e.g., 1e-06), the Hessian structure's impact is profound but is entirely mediated by the choice of `link_approx`.

1.  **The `bridge` link**

When using the bridge link approximation at low prior_precision, we uncover a nuanced story with a clear divergence between the model's calibration (ECE) and its predictive likelihood (NLL).

![bridge-hessian_ece](\blog_plots\summary_hessian_vs_precision_extremes_for_ece_bridge.png)

When we evaluate the models based on their ECE, a clear and consistent pattern emerges. The `kron` Hessian approximation, which captures correlations between weights in the model's final layer, consistently yields a lower (better) ECE than the simpler `diag` approximation. This holds true for both the ResNet18 and WideLeNet architectures and across all tested degrees of distribution shift.  This suggests the `bridge` approximation is stable enough to leverage the richer correlation information from `kron` to produce more reliable and well-calibrated confidence estimates.

From a likelihood (NLL) perspective, the model choice seems to make a difference. 

![bridge-hessian_nll](\blog_plots\model_specific_summary_hessian_vs_precision_for_nll_bridge.png)

For the deeper ResNet18 model (first row), capturing weight correlations is clearly beneficial for in-distribution and low-shift data. The `kron` Hessian provides a significantly lower (better) NLL, starting near-zero and outperforming the diag model up to a distribution shift of approximately 75°. This suggests that for ResNet18, the learned relationships between final-layer weights are important for making accurate predictions on familiar data. However, this reliance becomes a liability under severe distribution shift. Beyond 75°, the learned correlation structure becomes invalid and misleading, causing the kron model's performance to degrade sharply and fall below that of the more robust diag model. The `diag` approximation, by never relying on these complex correlations, degrades more gracefully, even if its initial performance is worse.

The shallower and wider WideLeNet model behaves differently. Here, the `kron` Hessian is not just better, it is absolutely essential. It achieves near-perfect NLL on ID data, while the diag model fails catastrophically from the start, posting an extremely high NLL of ~12. This indicates that for WideLeNet's architecture, the correlations in the final layer are fundamental to the model's predictive capabilities, ignoring them makes the model effectively useless. While the `kron` model's performance also degrades and its NLL skyrockets after a moderate shift, it remains vastly superior to the `diag` model across the entire spectrum of rotations. In this case, a degraded correlated posterior is still significantly better than no correlation at all.

2. **The `probit` link**

When we switch to the probit link, the situation becomes much more nuanced and depends entirely on the degree of distribution shift for both of the metrics.

![probit-hessian_ece](\blog_plots\summary_hessian_vs_precision_extremes_for_ece_probit.png)

![probit-hessian_nll](\blog_plots\summary_hessian_vs_precision_extremes_for_nll_probit.png)

For ID and low-shift data, the simpler `diag` Hessian is the clear winner across both metrics. It provides a significantly better NLL and, crucially, a better ECE. This outcome is likely tied to the nature of the probit function, which is highly sensitive to the variance of its input. In this low-regularization setting, the `kron` approximation appears to capture noisy, fine-grained curvature details. These details translate into unstable variance estimates that are detrimental to calibration and likelihood. The `diag` structure, by ignoring these complex correlations, provides a more robust and stable result, proving more effective when the data is close to the training distribution.

As the data shifts further into OOD (e.g., rotation angle > 60°), the performance of the two Hessian approximations diverges significantly. The `diag` model shows a consistent degradation in performance across both metrics as the shift increases: its NLL worsens, and its ECE also increases. In contrast, the `kron` model exhibits a different behavior. While its NLL remains consistently poor and largely flat, its calibration improves dramatically, with its ECE dropping sharply.
Around a 60° rotation, the kron model's ECE crosses below the diag model's, making it the better-calibrated model under severe distribution shifts. This happens because, in this high-error regime where predictions are failing, the richer curvature information from `kron`, even if noisy, provides a more honest and accurate picture of the model's rapidly increasing uncertainty. Although its predictions are poor (relatively high NLL), it correctly signals its low confidence, which results in better calibration. The simpler `diag` structure cannot capture this complex uncertainty landscape, so while its NLL is better, its calibration continues to degrade.

### The minor role of temperature scaling

Among the hyperparameters investigated, `temperature` scaling emerges as the one with the least influence on the overall results. Temperature scaling is a post-hoc calibration technique that "softens" or "sharpens" the final predictive probabilities by dividing the logits (the inputs to the final softmax) by a temperature value, T. Smaller values of T make the probability distribution sharper (more confident). While included as a standard calibration tool, its impact in the LA framework is consistently minimal.

Observing the ECE plots across different `temperature` settings (0.1, 0.5, and 1.0) reveals that this hyperparameter primarily acts as a minor vertical scaling factor on the performance curves, an effect that is often barely visible, as shown in this plot:

![temp-ece-nodiff](\blog_plots\hessian_comparison_by_temp_ece_context_prior0.01_linkbridge.png)

It does not fundamentally alter the shape of the curves, the relative performance of the `diag` and `kron` Hessians, or the critical crossover point where `kron` becomes better calibrated under high distribution shift.

The reason for this limited impact is its relationship with the other, more dominant hyperparameters. The predictive variance, which temperature scaling modulates, is already heavily determined by the interplay between `prior_precision` and `hessian_structure`. These parameters define the fundamental shape and scale of the posterior approximation. Temperature scaling is applied at the very end of the process, but if the underlying variance estimated by the Laplace approximation is already very large (high uncertainty) or very small (high confidence), a simple scalar division can only make marginal adjustments.

Interestingly, the `diag` Hessian appears to be slightly more sensitive to temperature changes than the `kron` Hessian. This is most visible at low-to-moderate shifts, where the diag curve shifts more noticeably with temperature. This could be because the simpler `diag` posterior produces a less extreme initial variance, giving the temperature parameter more "room" to have a discernible effect. The `kron` posterior, in contrast, often results in more extreme variance estimates (either very certain or very uncertain), leaving little for temperature scaling to modify.

![temp_ece](\blog_plots\hessian_comparison_by_temp_ece_context_prior0.01_linkprobit.png)