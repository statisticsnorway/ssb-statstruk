# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from statstruk import BaseModel

# %%
import re
def extract_vars(model_formula):
    """test
    """
    lhs, rhs = model_formula.split('~')
    x_vars = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', rhs))
    return(lhs, x_vars)



# %%
