# {{ml-boilerplate}}

# {{ml-boilerplate.description}}

## Project Organization

```
|   .gitignore
|   config.ps1                      <- Aliases for Python commands.
|   LICENSE                         <- Open-source license if one is chosen.
|   pyproject.toml                  <- Project configuration file with package metadata for {{ml-boilerplate.module_name}} and configuration for tools like black.
|   README.md                       <- The top-level README for developers using this project.
|   requirements.txt                <- The requirements file for reproducing the analysis/engineering environment, e.g. generated with `pip freeze > requirements.txt`.
|   setup.cfg                       <- Configuration file.
|           
+---data
|   +---external                    <- Data from third party sources.
|   |       .gitkeep
|   |       
|   +---interim                     <- Intermediate data that has been transformed. 
|   |       .gitkeep
|   |       
|   +---processed                   <- The final, canonical data sets for modeling.
|   |       .gitkeep
|   |       
|   \---raw                         <- The original, immutable data dump.
|           .gitkeep
|           
+---docs                            <- A default mkdocs project; see www.mkdocs.org for details.
|   |   .gitkeep
|   |   
|   \---mkdocs
|       |   mkdocs.yml
|       |   README.md
|       |   
|       \---docs
|               getting-started.md
|               index.md
|               
+---models                          <- Trained and serialized models, model predictions, or model summaries.
|       .gitkeep
|       
+---notebooks                       <- Jupyter notebooks. Naming convention is a number (for ordering), e.g. `1.0-explore`.
|       .gitkeep
|       
+---references                      <- Data dictionaries, manuals, and all other explanatory materials.
|       .gitkeep
|       
+---reports                         <- Generated analysis as HTML, PDF, LaTeX, etc.
|   |   .gitkeep
|   |   
|   \---figures                     <- Generated graphics and figures to be used in reporting.
|           .gitkeep
|           
\---{{ ml-boilerplate.module_name }} <- Source Code for use in this project.
    |   config.py
    |   __init__.py                 <- Makes {{ml-boilerplate.module_name}} a Python module.
    |   
    +---data                        <- Modules and scripts for data processing.
    |       dataset.py
    |       
    +---features                    <- Modules and scripts for feature engineering.
    |       features.py
    |       
    +---models                      <- Modules and scripts to run model training and inference.
    |       predict.py
    |       train.py
    |       __init__.py
    |       
    \---visualizations              <- Modules and scripts to create visualizations.
            plots.py
```

--------

# Dataset

# Installation

# Setup

# Contributors
