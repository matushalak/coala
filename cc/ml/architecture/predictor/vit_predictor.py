# TODO
# takes as input (masked) latents from all layers of encoder; 
# patchifies them to same resolution (conv) and concatenates along channels
# then applies a ViT-style transformer to predict masked latents at all layers
# un-patchifies via transposed conv