# Handwritten-Classification-Using-SVD

An SVD basis classification algorithm
Training: For the training set of known digits, compute the SVD of each set of
digits of one kind.

Classification: For a given test digit, compute its relative residual in all 10 bases. If one residual is significantly smaller than all the others, classify as that. Otherwise give up.
