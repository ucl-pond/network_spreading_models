from network_diffusion_model import NDM

class FKPP(NDM):
    def __init__(self, connectome_fname, gamma, t, x_0):
        super().__init__(self, connectome_fname, gamma, t, x_0)

    def logistic_model(self, x):
        return x*(1-x)

    def run_FKPP(self,x,alpha,dt):

        dx = alpha*self.NDM_step(H,x,dt) + (1-alpha)*self.logistic_model(x)*dt

        return x+dx