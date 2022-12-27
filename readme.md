# Design Patterns 

Design patterns generalized for reuse

# SQL

* sql/cte_example.py: a SQLAlchemy connection object is used to execute a SQL string joining two tables with common key


# Data Science

* data-science/fft.py: Numpy FFT is used in a dataclass to first zero-pad the signal then to calculate the Fast Fourier Transform corresponding coefficients and reconstructed signal with an option to ad-hoc bandpass filter the coefficients on an index-basis

* data-science/graph_laplacian.py: Scipy spatial distance matrix and Numpy linalg eigen decomposition are used in a semi-supervised spectral clustering dataclass

# Lib

* lib/utils.py: logging, get json enum key config, zip archive & delete directory

* lib/command_arguments.py: demonstrate splitting runtime parameters between command line interface and config file definitions
