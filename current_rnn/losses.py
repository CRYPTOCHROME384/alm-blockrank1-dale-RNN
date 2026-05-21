import torch as tch 
import numpy as np




class LossSparsity:
    def __init__(self, device):
        self.device = device
        self.l1_loss = tch.nn.L1Loss()

    def __call__(self, net):
        J_rec = net.J_rec
        total_loss = self.l1_loss(J_rec, tch.zeros(J_rec.shape).to(self.device))
        return total_loss

class SelectivityLoss:
    def __init__(self, rates, params_dict):
        self.dt = params_dict['dt']
        self.t_min_input = params_dict['t_min_input']
        self.t_max_input = params_dict['t_max_input']
        self.t_start = params_dict['t_start']
        self.t_go = params_dict['t_go']
        self.ind_delay = int((self.t_max_input- self.t_start)/self.dt)
        self.ind_sample = int((self.t_min_input - self.t_start)/self.dt)
        self.ind_go = int((self.t_go - self.t_start)/self.dt)
        self.d_ind = int(0.6/self.dt)
        self.compute_factors(rates)
        self.gram_schmidt(rates)
        self.cos =  tch.nn.CosineSimilarity(dim=0)
        self.mode_sample = 0
        self.mode_delay = 1
        self.mode_choice = 2
        self.alpha_delay = 1.5
        
    def __call__(self, net):
        #sample vectors
        net_sample_alm = net.decoder_alm[self.mode_sample,:]
        #delay vectors
        net_delay_alm = net.decoder_alm[self.mode_delay,:]
        #choice vectors
        net_choice_alm = net.decoder_alm[self.mode_choice,:]

        #losses
        loss_sample = (1 - self.cos(net_sample_alm, self.sample_alm))
        loss_delay = (1 - self.cos(net_delay_alm, self.delay_alm))
        loss_choice = (1 - self.cos(net_choice_alm, self.choice_alm))

        total_loss = (loss_sample + self.alpha_delay * loss_delay + loss_choice )/3.
        return total_loss

    def compute_factors(self, rates):
        self._sample_vectors(rates)
        self._delay_vectors(rates)
        self._choice_vectors(rates)

    def gram_schmidt(self, rates):
        self._sample_vectors(rates)
        self._delay_vectors(rates)
        self._choice_vectors(rates)
        vectors = tch.stack([self.sample_alm, self.delay_alm, self.choice_alm], dim=1) 
        # Perform QR decomposition for orthonormalization
        Q, R = tch.linalg.qr(vectors)

        # Q now contains the orthonormalized vectors
        orthonormal_vectors = Q.T  # Transpose to get them as a list

        # If you need them separately
        orthonormal_sample_alm, orthonormal_delay_alm, orthonormal_choice_alm = orthonormal_vectors
        self.sample_alm = tch.sign(R[0,0]) *  orthonormal_sample_alm # right increases
        self.delay_alm = tch.sign(R[1,1]) * orthonormal_delay_alm
        self.choice_alm = tch.sign(R[2,2]) * orthonormal_choice_alm
    
    def _sample_vectors(self, rates):
        y_alm = rates
        #calculating latent vectors
        vec_alm = y_alm[1,:, :] - y_alm[0, :, :]
        self.sample_alm = tch.mean(vec_alm[self.ind_delay-self.d_ind:self.ind_delay,:], axis=0)
        #self.sample_alm = tch.mean(vec_alm[self.ind_sample:self.ind_sample+self.d_ind,:], axis=0)

    def _delay_vectors(self, rates):
        y_alm = rates
        #calculating latent vectors
        vec_alm = y_alm[1,:, :] - y_alm[0, :, :]
        self.delay_alm = tch.mean(vec_alm[self.ind_go-self.d_ind:self.ind_go, :], axis=0)
    
    def _choice_vectors(self,rates):
        y_alm = rates
        #calculating latent vectors
        d_ind = int(0.4/self.dt)
        vec_alm = y_alm[1, :, :] - y_alm[0, :, :]
        self.choice_alm = tch.mean(vec_alm[self.ind_go:self.ind_go + d_ind, :], axis=0)
    


class SelectivitySingleTrialLoss:
    def __init__(self, rates, params_dict, indexes_neurons):
        self.dt = params_dict['dt']
        self.t_min_input = params_dict['t_min_input']
        self.t_max_input = params_dict['t_max_input']
        self.t_start = params_dict['t_start']
        self.t_go = params_dict['t_go']
        self.ind_delay = int((self.t_max_input- self.t_start)/self.dt)
        self.ind_sample = int((self.t_min_input - self.t_start)/self.dt)
        self.ind_go = int((self.t_go - self.t_start)/self.dt)
        self.d_ind = int(0.6/self.dt)
        self.compute_factors(rates)
        self.gram_schmidt(rates)
        self.mode_sample = 0
        self.mode_delay = 1
        self.mode_choice = 2
        self.indexes = indexes_neurons

    def __call__(self, net, rates_data, latents_trials):

        proj_sample_data, proj_delay_data, proj_choice_data = self.project_data(rates_data)
  

        #print(tch.mean(proj_delay_data), tch.mean(proj_delay_model))
        loss_sample = tch.mean(tch.square(proj_sample_data - latents_trials[:,:,self.mode_sample]))
        loss_delay = tch.mean(tch.square(proj_delay_data - latents_trials[:,:,self.mode_delay]))
        loss_choice = tch.mean(tch.square(proj_choice_data - latents_trials[:,:,self.mode_choice]))
        total_loss = (loss_sample + loss_delay + loss_choice )/3.
        return total_loss

    def compute_factors(self, rates):
        self._sample_vectors(rates)
        self._delay_vectors(rates)
        self._choice_vectors(rates)
    
    def project_data(self, rates_data):
        #losses
        proj_sample_data = rates_data @ self.sample_alm[self.indexes] 
        proj_delay_data = rates_data @ self.delay_alm[self.indexes]
        proj_choice_data = rates_data @ self.choice_alm[self.indexes] 
        return proj_sample_data, proj_delay_data, proj_choice_data

    def gram_schmidt(self, rates):
        self._sample_vectors(rates)
        self._delay_vectors(rates)
        self._choice_vectors(rates)
        vectors = tch.stack([self.sample_alm, self.delay_alm, self.choice_alm], dim=1) 
        # Perform QR decomposition for orthonormalization
        Q, R = tch.linalg.qr(vectors)

        # Q now contains the orthonormalized vectors
        orthonormal_vectors = Q.T  # Transpose to get them as a list

        # If you need them separately
        orthonormal_sample_alm, orthonormal_delay_alm, orthonormal_choice_alm = orthonormal_vectors
        self.sample_alm = tch.sign(R[0,0]) *  orthonormal_sample_alm # right increases
        self.delay_alm = tch.sign(R[1,1]) * orthonormal_delay_alm
        self.choice_alm = tch.sign(R[2,2]) * orthonormal_choice_alm

    def _sample_vectors(self, rates):
        y_alm = rates
        #calculating latent vectors
        vec_alm = y_alm[1,:, :] - y_alm[0, :, :]
        self.sample_alm = tch.mean(vec_alm[self.ind_delay-self.d_ind:self.ind_delay,:], axis=0)

    def _delay_vectors(self, rates):
        y_alm = rates
        #calculating latent vectors
        vec_alm = y_alm[1,:, :] - y_alm[0, :, :]
        self.delay_alm = tch.mean(vec_alm[self.ind_go-self.d_ind:self.ind_go, :], axis=0)
    
    def _choice_vectors(self,rates):
        y_alm = rates
        #calculating latent vectors
        d_ind = int(0.4/self.dt)
        vec_alm = y_alm[1, :, :] - y_alm[0, :, :]
        self.choice_alm = tch.mean(vec_alm[self.ind_go:self.ind_go + d_ind, :], axis=0)
    

class LossAverageTrials:
    def __init__(self):
        self.alpha = 0.1
        
    def __call__(self, av_trials_data, rates_alm):
        #calculating losses
        total_loss = tch.mean(tch.square(av_trials_data- rates_alm))
        return total_loss

class LossAverageTime:
    def __init__(self, params_dict):
        self.t_min_input = params_dict['t_min_input']
        self.t_max_input = params_dict['t_max_input']
        self.t_start = params_dict['t_start']
        self.t_go = params_dict['t_go']
        self.dt = params_dict['dt']

        self.t_after_go = 0.4
        t_sample_s = self.t_min_input - self.t_start 
        t_sample_e = self.t_max_input - self.t_start
        t_delay_e = self.t_go - self.t_start
        t_response = self.t_after_go - self.t_start

        self.ind_sample_s =  int(t_sample_s/self.dt)
        self.ind_sample_e = int(t_sample_e/self.dt)
        self.ind_delay_e1 = int(t_delay_e/(2 * self.dt))
        self.ind_delay_e2 = int(t_delay_e/self.dt)
        self.ind_response = int(t_response/self.dt)

    def __call__(self, rates_data, rates_model):
        # mean spikes across neurons
        av_time_model = self.compute_time_averages_4background(rates_model)
        av_time_data = self.compute_time_averages_4background(rates_data)
        #calculating losses
        total_loss = tch.mean(tch.square(av_time_data - av_time_model))
        return total_loss

    def compute_time_averages_4(self, rates):
        average_sample = tch.mean(rates[:, self.ind_sample_s:self.ind_sample_e, :], axis = 1)  
        average_delay1 = tch.mean(rates[:, self.ind_sample_e:self.ind_delay_e1, :], axis = 1) 
        average_delay2 = tch.mean(rates[:, self.ind_delay_e1:self.ind_delay_e2,:], axis = 1) 
        average_response = tch.mean(rates[:, self.ind_delay_e2:self.ind_response, :], axis = 1) 
        av_time_model = tch.stack((average_sample, average_delay1, average_delay2, average_response)) #condition, trials, neurons
        return av_time_model
    def compute_time_averages_3(self, rates):
        average_sample = tch.mean(rates[:, self.ind_sample_s:self.ind_sample_e, :], axis = 1)  
        average_delay = tch.mean(rates[:, self.ind_sample_e:self.ind_delay_e2, :], axis = 1) 
        average_response = tch.mean(rates[:, self.ind_delay_e2:self.ind_response, :], axis = 1) 
        av_time_model = tch.stack((average_sample, average_delay, average_response)) #condition, trials, neurons
        return av_time_model
    def compute_time_averages_4background(self, rates):
        average_background = tch.mean(rates[:, 0:self.ind_sample_s, :], axis = 1)  
        average_sample = tch.mean(rates[:, self.ind_sample_s:self.ind_sample_e, :], axis = 1)  
        average_delay = tch.mean(rates[:, self.ind_sample_e:self.ind_delay_e2, :], axis = 1) 
        average_response = tch.mean(rates[:, self.ind_delay_e2:self.ind_response, :], axis = 1) 
        av_time_model = tch.stack(( average_background, average_sample, average_delay, average_response)) #condition, trials, neurons
        return av_time_model

class LossAllSpikes:
    def __init__(self, indexes_neurons):
        self.indexes_neurons = indexes_neurons

    def __call__(self, spikes, rates):
        # mean spikes across neurons
        total_loss = tch.mean(tch.square(spikes- rates))
        return total_loss
    


class LossAverageNeurons:
    def __init__(self, indexes_neurons):
        self.indexes_neurons = indexes_neurons

    def __call__(self, av_neurons_data, rates):
        # mean spikes across neurons
        av_neurons_model = self.compute_average_neurons(rates)
        total_loss = tch.mean(tch.square(av_neurons_data - av_neurons_model))
        return total_loss
    
    def compute_average_neurons(self, rates):
        av_neurons_model = tch.mean(rates[:,:, self.indexes_neurons], axis = 2)#spikes per s
        return av_neurons_model

class LossOrthogonality:
    def __init__(self, device):
        self.device = device
        
    def __call__(self, cov_matrix):
        cov_alm = cov_matrix['cov_alm']
        eye_alm = tch.eye(cov_alm.shape[0]).to(self.device).float()
        total_loss = tch.mean(tch.square(cov_alm - eye_alm))
        return total_loss