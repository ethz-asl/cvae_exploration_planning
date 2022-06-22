# CNN-based Gain Predictor in Pytorch

### Map encoder

input: 1×25×25 0/1/2 local map
|
three 5 × 5 convolutions and a max-pooling layer
|
output: 1×16 latent vector

### Pose encoder

input: 1×4 pose (dx, dy, cosphi, sinphi)
|
two dense layer
|
output: 1×16 latent vector

### Gain predictor

input: 1×(16+16)
|
two dense layer
|
output: 1 