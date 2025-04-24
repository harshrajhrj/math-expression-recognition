# Files
**checkpoint_one**

epoch_vs_loss_one.txt
* No batch normalization
* No weight initialization (xavier, kaiming)
* No convergence techniques
* No softmax
* No variations
* Trained for $30$ epochs
* Trained with $11000$ samples
* Trained with batch_size = $64$
* Learning rate = $1e-3$ or $0.001$
* Scheduling learning rate
    * step size = $10$
    * gamma = $0.1$
* No accuracy

---
**checkpoint_two**

epoch_vs_loss_two.txt
* No batch normalization
* Kaiming Normal Initialization
* No convergence techniques
* No softmax
* No variations
* Trained for $15$ epochs
* Trained with $15000$ samples
* Trained with batch_size = $32$
* Learning rate = $0.001$
* Scheduling learning rate
    * step size = $5$
    * gamma = $0.1$
* $97.8\%$ accuracy