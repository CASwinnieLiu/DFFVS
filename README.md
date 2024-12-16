# A Novel Manifold Optimization Algorithm with the Dual Function and a Fuzzy Valuation Step

### Introduction
This is a PyTorch implementation of our methods introduced in the following paper ...

- Weiping Liu et al.,A Novel Manifold Optimization Algorithm with the Dual Function and a Fuzzy Valuation Step,TFS,2024 



### How-to-use

#### Instructions
####
product optimizer
  - Import product_optimizer.py
    - `from DFFVS_product_optimizer import DFFVS_p`
  - Initialize just like an optimizer
    - ` optimizer= DFFVS_p(net.parameters())`
complex product optimizer
  - Import complex_product_optimizer.py
    - `from DFFVS_complex_product_optimizer import DFFVS_Cp`
  - Initialize just like an optimizer
    - ` optimizer= DFFVS_Cp(net.parameters())`
complex grassmann optimizer
  - Import complex_grassmann_optimizer.py
    - `from DFFVS_complex_grassmann_optimizer import DFFVS_Cg`
  - Initialize just like an optimizer
    - ` optimizer= DFFVS_Cg(net.parameters())`
###bgw
###
   - Import bgw.py
    - `from DFFVS_bgw import bgw`
  - Initialize just like an optimizer
    - ` lr = bgw(net.parameters())`
