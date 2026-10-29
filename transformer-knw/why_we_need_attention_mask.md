### we need an attention mask to control which tokens a model is allowed to pay attention to during the attention calculation.

Problem
```text
The mask never reaches the model. In embeddings.py:74 you keep only       
     tok.encode(...)[0], so the mask is thrown away. Attention needs it. Add a 
     small helper that returns two tensors, ids (B, T) and mask (B, T)
```
Solution
```text

```

