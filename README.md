# Multithreaded GPT

Multithreaded GPT is an experimental extension of GPT models, designed to explore parallel processing in transformers by applying unique sets of weights for each token in a sequence. This project builds on the transformer architecture from [Andrej Karpathy's nanoGPT course](https://github.com/karpathy/nanoGPT) and introduces a custom `MultiLinear` layer to handle token-specific weights for more intelligent processing.

## Overview

In traditional transformer models, a single set of weights is applied across all tokens in a sequence. This allows efficient parallelism, but all tokens undergo the same transformation. Multithreaded GPT modifies this behavior by using a unique set of weights for each token through the custom `MultiLinear` layer:

```python
class MultiLinear(nn.Module):
    """Custom Linear layer with separate weights for each time-step"""
    
    def __init__(self, in_features, out_features, num_timesteps, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_timesteps = num_timesteps
        self.weight = nn.Parameter(torch.randn(num_timesteps, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(num_timesteps, out_features)) if bias else None

    def forward(self, x):
        B, T, C = x.shape
        assert T <= self.num_timesteps, f"Input time steps {T} exceed layer's configured time steps {self.num_timesteps}"
        out = torch.einsum('btc,tcd->btd', x, self.weight)
        if self.bias is not None:
            out += self.bias
        return out
```

### Key Differences
- **Separate Weights per Token**: Each token in the input sequence has its own set of weights, increasing the model's capacity to handle complex tasks differently for each token.
- **Parallel Processing**: This design aims to leverage the additional weights for parallel token processing, potentially improving performance on certain tasks.
- **Aggregation Layer**: After processing each token independently, the output is passed through a 'cap' layer that aggregates the information across all tokens. This could be replaced with other aggregation strategies for further experimentation. note that by default weigth are initialized to return last token input as output.

## Motivation

The main goal of Multithreaded GPT is to investigate if giving each token its own processing path can lead to more intelligent and diverse behavior when solving complex problems. While early experiments have not yielded significant improvements, there is potential for future research, particularly in parallel task processing and alternative aggregation methods.

## Results

Thus far, the experiments with Multithreaded GPT have not demonstrated significant performance improvements. However, further experimentation with different tasks, larger datasets, or alternative aggregation strategies may yield more promising results.

## Future Work

- **Different Aggregation Methods**: The current 'cap' layer aggregates token outputs using a large linear layer. Other techniques, like attention-based pooling, could be explored.
- **Parallel Task Processing**: Investigating how this architecture could handle more complex, multi-threaded tasks.
- **Larger Experiments**: Testing the model on larger datasets or more challenging tasks to evaluate its effectiveness.

## Contributions

Feel free to contribute by experimenting with new aggregation techniques, testing on different tasks, or optimizing the parallelism in token processing.

---

Let me know if you'd like to refine any part or add more specific instructions!
