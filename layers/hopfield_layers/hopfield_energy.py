__author__="a-kore"
__editedby__="Jumperkables"
"""
Hopfield Energy implemented by a generous soul: https://github.com/a-kore/hopfield-energy-demo/blob/main/hopfield_energy_demo.ipynb
Thanks to a-kore for sharing in the official GitHub issues section
"""


"""
@Jumperkables:
Explanation:
    - If I have understood correctly, direct calculation of the energy function doesnt happen in the network or these layers
    - The update rule is instead shown to minimise this theoretical energy function by proxy
    - Directly applying this energy function on the raw state and stored patterns will give you some understanding how well they are currently learned by the Hopfield layer
 
My Idea:
    - I can check that my training was successful by applying an altered term
"""
def get_energy(R, Y, beta):
    """
    R:      Raw state patterns
    Y:      Raw stored patterns
    beta:   Scaling number
    """
    lse = -(1.0/beta)*torch.logsumexp(beta*(torch.bmm(R, Y.transpose(1,2))), dim=2) # -lse(beta, Y^T*R)
    lnN = (1.0/beta)*torch.log(torch.tensor(Y.shape[1], dtype=float)) # beta^-1*ln(N)
    RTR = torch.bmm(R, R.transpose(1,2)) # R^T*R
    M = 0.5*((torch.max(torch.linalg.norm(Y, dim=2), dim=1))[0]**2.0) # 0.5*M^2  *very large value*
    energy = lse + lnN + RTR + M
    return energy
