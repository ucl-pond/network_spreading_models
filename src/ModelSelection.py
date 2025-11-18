import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

class ModelSelection:
    """
    Metrics for model selection.
    Dummy example for 5 models each having 100 obeservations using AICc criterion: 
        >>> residuals = np.random.normal(size=(100,5))
        >>> results = ModelSelection([3,3,4,2,3], 'AICc')(residuals)
    """
    
    def __init__(self, n_dof, criterion='AICc', model_names=None, plot=True):
        """
        Inputs
        ----------
        n_dof : list of int, length n_models
            Number of degrees of freedom of each model.
        criterion : str, optional
            Model selection criterion: 
                - 'AIC': Akaike Information Criterion
                - 'AICc': AIC corrected
                - 'BIC': Bayesian Information Criterion
            Default is AICc.
        model_names : list of str, length n_models, optional
            Name of each model.
        plot : bool, optional
            Show plots or not. Default is True.
        """
        
        self.criterion = criterion
        self.n_dof = n_dof
        self.model_names = model_names
        self.plot = plot
        self.n_models = len(n_dof)
        

    def __call__(self, residuals):
        """
        Inputs
        ----------
        residuals: array, size (n_obs, n_models)
            residuals from models fitting.
            
        Outputs
        ----------      
        C: list, length n_models
            Criterion for each model.        
        C_w: list, length n_models
            Criterion weight for each model 
            (probability of this model being the "true" model 
             assuming the "true" model is part of the candidates).
        """
        
        if self.n_models != residuals.shape[1]:
            sys.exit("The length of n_dof and the number of columns of residuals should be the same (number of models).")
        if self.model_names is None:
            self.model_names = []
            for m in range(self.n_models):
                self.model_names += ['model ' + str(m+1)]
                    
                    
        n_obs = residuals.shape[0]
        
        ss_res = np.sum(residuals ** 2, 0)
        loglikelihood = (-1/2) * n_obs * np.log(ss_res / n_obs) # assumes Gaussian distribution of the residuals
        
        C = np.full(self.n_models, np.nan)
        C_w = np.full(self.n_models, np.nan)
        
        for m in range(self.n_models):        
            C[m] = self.get_criterion(loglikelihood[m], n_obs, self.n_dof[m])
        
        Cmin = np.min(C)
        for m in range(self.n_models):             
            C_w[m] = self.get_criterion_weight(C[m], Cmin)
        C_w /= np.sum(C_w)
        
        rank = np.zeros(self.n_models, dtype=int)
        ind = np.argsort(C)
        for i in range(self.n_models):
            rank[ind[i]] = i+1
        
        if self.plot:
            self.plot_stuff(residuals, C, C_w)
                    
        results = pd.DataFrame(data = {'Model': self.model_names,
                                  self.criterion: C,
                                  self.criterion + ' weights': C_w,
                                  'rank': rank})
            
        return results
            
            
    def get_criterion(self, loglikelihood, n_obs, n_dof):
        
        # Akaike information criterion (AIC)
        if self.criterion in ('AIC', 'AICc'):
            C = 2 * n_dof - 2*loglikelihood
            
            # AIC corrected (AICc)
            if self.criterion == 'AICc':
                C += 2*n_dof*(n_dof+1)/(n_obs-n_dof-1)
                
        # Bayesian information criterion (BIC)
        elif self.criterion == 'BIC':
            C = n_dof * np.log(n_obs) - 2*loglikelihood
            
        return C
                
        
    def get_criterion_weight(self, C, Cmin):
        
        C_w = np.exp((Cmin - C)/2)
            
        return C_w
    
            
    def plot_stuff(self, residuals, C, C_w):
    
        cols = plt.cm.get_cmap('Dark2')
        
        fig, axes = plt.subplots(1, 3, figsize=(7,4), dpi=400)
        
        # squared errors
        se_res = residuals ** 2
        axes[0].plot(np.transpose(se_res), color=[0.9]*3)
        for m in range(self.n_models):
            bp = axes[0].boxplot(se_res[:,m], positions=[m], widths=0.5,
                                 patch_artist=True, flierprops=dict(marker='.', markeredgecolor=cols(m)[0:3]))  
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=cols(m)[0:3] + (1,))  
            for patch in bp['boxes']:
                patch.set(facecolor=cols(m)[0:3] + (0.3,)) 
        axes[0].set_ylim(bottom=0)
        axes[0].set_title('Squared Errors')
            
        # criterion
        for m in range(self.n_models):
            axes[1].plot([m-0.25,m+0.25],[C[m],C[m]], color=cols(m)[0:3], linewidth=3)
        axes[1].set_title(self.criterion)
        
        # criterion weights
        for m in range(self.n_models):
            axes[2].bar(height=C_w[m], x=m, color=cols(m)[0:3], width=0.5)
        axes[2].set_title(self.criterion + ' weights')
        axes[2].set_ylim(top=1)
           
        for p in range(3):
            axes[p].set_xticks(range(self.n_models))
            axes[p].set_xticklabels(self.model_names, rotation=45, ha='right')
            
        plt.suptitle('Model Selection Metrics')
        plt.tight_layout()
        plt.show()
        

           
#residuals = np.random.normal(size=(100,5))
#results = ModelSelection([3,3,4,2,3], 'AICc')(residuals)
#print(results)


