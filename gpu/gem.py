import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time 
import random

class nnHMRF_LIS:
    def __init__(self, x, device, savepath):
        torch.set_default_dtype(torch.float64)
        self.SAVE_DIR = savepath
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
                
        self.KERNEL_PATH = os.path.join(os.getcwd(), 'kernel.txt')

        self.params = {'B': 0,
                       'H': 0,
                       'B_prev': 0,
                       'H_prev': 0,
                       'pL': torch.tensor([0.7,0.3]),
                       'muL': torch.tensor([-1.0,3.0]),
                       'sigmaL2': torch.tensor([2.0,2.0]),
                       'pL_prev': torch.zeros(2),
                       'muL_prev': torch.zeros(2),
                       'sigmaL2_prev': torch.zeros(2)
                       }
        self.const = {'a': 1,  # for penalized likelihood for L >= 2
                      'b': 2,
                      'delta': 1e-3,
                      'maxIter': 1,
                      'eps1': 1e-3,
                      'eps2': 1e-2,
                      'eps3': 1e-3,
                      'alpha': 1e-4,
                      'burn_in': 2,
                      'num_samples': 2,
                      'L': 2,
                      'newton_max': 3,
                      'fdr_control': 0.1,
                      'tiny': 1e-8
                      }
        self.device = device
        self.x = torch.from_numpy(x).to(self.device)
        self.data_dim = self.x.shape
        self.gamma = torch.zeros(self.x.shape).to(self.device)  # P(theta | x)
        self.init = torch.zeros(self.x.shape).to(self.device)  # initial theta value for gibb's sampling
        self.H_x = torch.zeros(2).to(self.device)
        self.H_mean = torch.zeros(2).to(self.device)
        self.log_sum = torch.zeros(1).to(self.device)
        self.U = torch.zeros(2).to(self.device)
        self.I_inv = torch.zeros((2, 2)).to(self.device)
        self.H_iter = torch.zeros((self.const['num_samples'], 2)).to(self.device)
        self.H_x_iter = torch.zeros((self.const['num_samples'], 2)).to(self.device)
        self.convergence_count = 0
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        init_prob = torch.full(self.data_dim, 0.5)
        self.init = torch.bernoulli(init_prob).to(self.device)  # p = 0.5 to choose 1
        
        self.start = time.time()
        self.end = 0
        
        self.white_map = torch.from_numpy(np.indices(self.data_dim).sum(axis=0) % 2).type(torch.FloatTensor)
        self.white_map = self.white_map.to(self.device)
        ones = torch.ones(self.data_dim).to(self.device)
        self.black_map = torch.logical_xor(self.white_map, ones).type(torch.FloatTensor)
        self.black_map = self.black_map.to(self.device)

    def gem(self):
        '''
        E-step: Compute the following
        phi: (6), (7), (8)
        varphi: (10), (11)
        Iterate until convergence using delta (13)
        '''
        # compute first term in hs(t) = h(t) - .... equation
        mu_0 = 0
        sigma0_sq = 1
        const_0 = 1 / torch.sqrt(torch.tensor([2 * torch.pi * sigma0_sq])).to(self.device)
        ln_0 = torch.log(const_0 * torch.exp(-(torch.pow(self.x - mu_0, 2)) / (2 * sigma0_sq))).to(self.device)
        
        for t in range(self.const['maxIter']):
            print("")
            print("GEM iter: ", t)
            print("B:  " + str(self.params['B']) + " B_prev: " + str(self.params['B_prev']))
            print("H:  " + str(self.params['H']) + " H_prev: " + str(self.params['H_prev']))
            print("pL:  " + str(self.params['pL']) + " B_prev: " + str(self.params['pL_prev']))
            print("muL:  " + str(self.params['muL']) + " muL_prev: " + str(self.params['muL_prev']))
            print("sigmaL2:  " + str(self.params['sigmaL2']) + " sigmaL2_prev: " + str(self.params['sigmaL2_prev']))
            # varphi: (10), (11)
            self.params['B_prev'] = self.params['B']
            self.params['H_prev'] = self.params['H']
            gamma_sum = 0  # !!!!!!!!!!! need to compute

            # calculate hs(t)
            hs = torch.zeros(self.x.shape)
            ln_1 = 0
            for l in range(self.const['L']):
                ln_1 += ((self.params['pL'][l]) / (torch.sqrt(2 * torch.pi * self.params['sigmaL2'][l]))) \
                        * torch.exp(-(torch.pow(self.x - self.params['muL'][l], 2)) / (2 * self.params['sigmaL2'][l]))
            ln_1 = torch.log(ln_1)

            hs = self.params['H'] - ln_0 + ln_1

            # Gibb's sampler to generate theta and theta_x
            theta, self.H_mean, self.H_iter, exp_sum, g = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], self.params['H'], self.params['B'])
            theta_x, self.H_x, self.H_x_iter, exp_x_sum, self.gamma = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], hs, self.params['B'])
            
            # Monte carlo averages
            self.H_mean /= self.const['num_samples']
            self.H_x /= self.const['num_samples']
            self.gamma /= self.const['num_samples']
            gamma_sum = torch.sum(self.gamma)
            
            # compute log factor for current varphi
            # Because np.exp can overflow with large value of exp_sum, a modification is made
            # where we subtract by max_val. Same is done for log_sum_next
            self.log_sum = torch.zeros(1)
            max_val = torch.amax(exp_sum)
            self.log_sum = torch.sum(torch.exp(exp_sum - max_val))
            self.log_sum = torch.log(self.log_sum) + max_val
            # compute U, I
            # U --> 2x1 matrix, I --> 2x2 matrix, trans(U)*inv(I)*U = 1x1 matrix
            self.U = self.H_x - self.H_mean
            self.I_inv = self.compute_I_inverse(self.const['num_samples'], self.H_iter, self.H_mean)
            # phi: (6), (7), (8)
            for l in range(self.const['L']):
                self.params['muL_prev'][l] = self.params['muL'][l]
                self.params['sigmaL2_prev'][l] = self.params['sigmaL2'][l]
                self.params['pL_prev'][l] = self.params['pL'][l]

            f_x = torch.zeros(self.x.shape).to(self.device)
            omega_sum = torch.zeros(self.const['L']).to(self.device)
            omega = torch.empty((2, *self.data_dim)).to(self.device)

            for l in range(self.const['L']):
                f_x += self.params['pL'][l] \
                                   * self.norm(self.x, self.params['muL'][l], self.params['sigmaL2'][l])
            for l in range(self.const['L']):
                omega[l] = self.gamma * self.params['pL'][l] / f_x \
                                       * self.norm(self.x, self.params['muL'][l], self.params['sigmaL2'][l])
                omega_sum[l] = torch.sum(omega[l])

            # t+1 update
            for l in range(self.const['L']):
                self.params['muL'][l] = self.sum_muL(omega[l], self.x) / omega_sum[l]
                self.params['sigmaL2'][l] = (2.0 * self.const['a'] + self.sum_sigmaL2(omega[l], self.x, self.params['muL'][l])) \
                                            / (2.0 * self.const['b'] + omega_sum[l])
                self.params['pL'][l] = omega_sum[l] / gamma_sum
                
            # find varphi(t+1) that satisfies Armijo condition and compute Q2(t+1) - Q2(t)
            self.computeArmijo()
            
            # check for parameter convergence (13) using phi and varphi
            if self.converged():
                self.convergence_count += 1
                if self.convergence_count == self.const['newton_max']:
                    print("converged at: ", t)
                    break
            else:
                self.convergence_count = 0
             
            self.end = time.time()
            print("time elapsed:", self.end - self.start)
        
        # Save B, H, mu, sigma, pl, gamma
        params_directory = os.path.join(self.SAVE_DIR, 'params.txt')
        with open(params_directory, 'w') as outfile:
            for key, value in self.params.items():
                outfile.write(str(key) + ': ' + str(value) + '\n')

        gamma_directory = os.path.join(self.SAVE_DIR, 'gamma_shmrf.npy')
        np.save(gamma_directory, self.gamma.cpu().numpy())

    def computeArmijo(self):
        lambda_m = 1
        diff = torch.zeros(2)
        # eq(11) satisfaction condition
        while True:
            delta = torch.matmul(self.I_inv, self.U)
            self.params['B'] = self.params['B_prev'] + lambda_m * delta[0]
            self.params['H'] = self.params['H_prev'] + lambda_m * delta[1]

            # check for alternative criterion when delta_varphi is too small for armijo convergence
            # In practice, the Armijo condition (11) might not be satisfied
            # when the step length is too small
            diff[0] = torch.abs(torch.tensor([self.params['B'] - self.params['B_prev']])) / (
                        torch.abs(torch.tensor([self.params['B_prev']])) + self.const['eps1'])
            diff[1] = torch.abs(torch.tensor([self.params['H'] - self.params['H_prev']])) / (
                        torch.abs(torch.tensor([self.params['H_prev']])) + self.const['eps1'])
            max = torch.amax(diff)
            if max < self.const['eps3']:
                print("alternative condition")
                self.params['B'] = self.params['B_prev']
                self.params['H'] = self.params['H_prev']
                break
            
            armijo = self.const['alpha'] * lambda_m * (torch.matmul(torch.transpose(self.U, 0, -1), torch.matmul(self.I_inv, self.U)))
            # use Gibbs sampler again to compute log_factor for delta_Q2 approximation
            log_factor = self.computeLogFactor()

            deltaQ2 = (self.params['B'] - self.params['B_prev']) * self.H_x[0] \
                      + (self.params['H'] - self.params['H_prev']) * self.H_x[1] + log_factor
            if deltaQ2 > 0:
                print("Q2: INCREASING")
            else:
                print("Q2: *********************************")

            if deltaQ2 >= armijo:
                break
            else:
                lambda_m /= 2
        return

    def computeLogFactor(self):
        theta_next, H_next_mean, H_next_iter, exp_sum_next, g = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], self.params['H'], self.params['B'])
        # compute log sum for varphi_next
        max_val = torch.amax(exp_sum_next)
        log_sum_next = torch.sum(torch.exp(exp_sum_next - max_val))
        log_sum_next = torch.log(log_sum_next) + max_val
        result = log_sum_next - self.log_sum
        return result

    def converged(self):
        # we have 8 parameters
        # B, H, pl[0], pl[1], mul[0], mul[1], sigmal[0], sigmal[1]
        diff = torch.zeros(8)
        diff[0] = torch.abs(torch.tensor([self.params['B']-self.params['B_prev']])) / (torch.abs(torch.tensor([self.params['B_prev']])) + self.const['eps1'])
        diff[1] = torch.abs(torch.tensor([self.params['H']-self.params['H_prev']])) / (torch.abs(torch.tensor([self.params['H_prev']])) + self.const['eps1'])
        diff[2] = torch.abs(torch.tensor([self.params['pL'][0]-self.params['pL_prev'][0]])) / (torch.abs(torch.tensor([self.params['pL_prev'][0]])) + self.const['eps1'])
        diff[3] = torch.abs(torch.tensor([self.params['pL'][1]-self.params['pL_prev'][1]])) / (torch.abs(torch.tensor([self.params['pL_prev'][1]])) + self.const['eps1'])
        diff[4] = torch.abs(torch.tensor([self.params['muL'][0]-self.params['muL_prev'][0]])) / (torch.abs(torch.tensor([self.params['muL_prev'][0]])) + self.const['eps1'])
        diff[5] = torch.abs(torch.tensor([self.params['muL'][1]-self.params['muL_prev'][1]])) / (torch.abs(torch.tensor([self.params['muL_prev'][1]])) + self.const['eps1'])
        diff[6] = torch.abs(torch.tensor([self.params['sigmaL2'][0]-self.params['sigmaL2_prev'][0]])) / (torch.abs(torch.tensor([self.params['sigmaL2_prev'][0]])) + self.const['eps1'])
        diff[7] = torch.abs(torch.tensor([self.params['sigmaL2'][1]-self.params['sigmaL2_prev'][1]])) / (torch.abs(torch.tensor([self.params['sigmaL2_prev'][1]])) + self.const['eps1'])

        max = torch.amax(diff)
        if max < self.const['eps2']:
            return True

        return False
        
    def gibb_sampler(self, burn_in, n_samples, h, beta):
        # initialize labels
        init_prob = torch.full(self.data_dim, 0.5)
        theta = torch.bernoulli(init_prob).to(self.device)  # p = 0.5 to choose 1
        kernel = torch.from_numpy(np.loadtxt(self.KERNEL_PATH).reshape((3,3,3))).float()
        kernel = torch.unsqueeze(kernel, 0)
        kernel = torch.unsqueeze(kernel, 0)
        kernel = kernel.to(self.device)
        H_iter = torch.zeros((self.const['num_samples'], 2)).to(self.device)
        H_mean = torch.zeros(2).to(self.device)
        exp_sum = torch.zeros(self.const['num_samples']).to(self.device)
        gamma = torch.zeros(self.x.shape).to(self.device)
        
        iteration = 0
        iter_burn = 0
        while iteration < n_samples:
            # calculate sum_nn
            # sum of neighboring pixels is a convolution operation
            theta = theta.type(torch.FloatTensor).to(self.device)
            theta = torch.unsqueeze(theta, 0)
            theta = torch.unsqueeze(theta, 0)
            sum_nn = torch.nn.functional.conv3d(theta, kernel, stride=1, padding=1).squeeze()
            # calculate exp_sum_nn
            numerator = torch.exp(sum_nn.type(torch.DoubleTensor).to(self.device) * beta + h)

            conditional_distribution = numerator / (1 + numerator)
            # then put it into bernoulli

            white = torch.bernoulli(conditional_distribution)
            white = torch.logical_and(white, self.white_map).type(torch.FloatTensor).to(self.device)

            black_label = torch.logical_and(theta.squeeze(), self.black_map).type(torch.FloatTensor).to(self.device)
            theta = torch.logical_or(black_label, white).type(torch.FloatTensor).to(self.device)
            theta = torch.unsqueeze(theta, 0)
            theta = torch.unsqueeze(theta, 0)
            sum_nn = torch.nn.functional.conv3d(theta, kernel, stride=1, padding=1).squeeze().to(self.device)

            # calculate exp_sum_nn
            numerator = torch.exp(sum_nn.type(torch.DoubleTensor).to(self.device) * beta + h)
            conditional_distribution = numerator / (1 + numerator)
            # then put it into bernoulli
            black = torch.bernoulli(conditional_distribution).to(self.device)
            black = torch.logical_and(black, self.black_map).type(torch.FloatTensor).to(self.device)
            theta = torch.logical_or(black, white).type(torch.FloatTensor).to(self.device)
            sum_nn = sum_nn.to(self.device)
            if iter_burn < burn_in:
                iter_burn += 1
            else:
                H_mean[0] += torch.sum(theta * sum_nn) #******************************* CHECK CORRECTNESS IN THESE SIGMAS
                H_mean[1] += torch.sum(theta)
                H_iter[iteration][0] = torch.sum(theta * sum_nn)
                H_iter[iteration][1] = torch.sum(theta)
              
                exp_sum[iteration] = -1 * self.H_iter[iteration][0] * self.params['B'] - self.H_iter[iteration][1] * self.params['H']
                gamma += theta
                iteration += 1
        return theta, H_mean, H_iter, exp_sum, gamma

    def compute_I_inverse(self, num_samples, H, H_mean):
        I = torch.zeros((2, 2)).to(self.device)
        for i in range(num_samples):
            I[0][0] += (H[i][0] - H_mean[0]) * (H[i][0] - H_mean[0])
            I[0][1] += (H[i][1] - H_mean[1]) * (H[i][0] - H_mean[0])
            I[1][0] += (H[i][1] - H_mean[1]) * (H[i][0] - H_mean[0])
            I[1][1] += (H[i][1] - H_mean[1]) * (H[i][1] - H_mean[1])

        I /= num_samples - 1  # n-1

        determinant = I[1][1] * I[0][0] - I[0][1] * I[1][0]
        if determinant > 0:
            return torch.inverse(I)
        else:
            # identity matrix
            return torch.eye(2).to(self.device)

    def norm(self, x, mu, sigma2):
        return torch.exp(-1 * (x - mu) * (x - mu) / (2 * sigma2)) / (torch.sqrt(2 * torch.pi * sigma2))
    
    def sum_muL(self, omega, x):
        return torch.sum(omega * x)

    def sum_sigmaL2(self, omega, x, muL):       
        return torch.sum(omega * torch.pow(x - muL, 2))
    

    def p_lis(self, gamma_1, threshold=0.1, label=None, savepath=None):
        """
        If groundtruth is specified (label is not None), then returns fdr, fnr, atp, signal_lis
        if groundtruth is not specified, saves LIS and gamma to savepath
        """
        gamma_1 = gamma_1.ravel()
        dtype = [('index', int), ('value', float)]
        size = gamma_1.shape[0]

        # flip
        lis = np.zeros(size, dtype=dtype)
        lis[:]['index'] = np.arange(0, size)
        lis[:]['value'] = 1-gamma_1 

        # get k
        lis = np.sort(lis, order='value')
        cumulative_sum = np.cumsum(lis[:]['value'])
        k = np.argmax(cumulative_sum > (np.arange(len(lis)) + 1)*threshold)

        signal_lis = np.zeros(size)
        signal_lis[lis[:k]['index']] = 1

        if savepath is not None:
            np.save(os.path.join(savepath, 'gamma.npy'), gamma_1)
            np.save(os.path.join(savepath, 'lis.npy'), signal_lis)

        if label is not None:
            # GT FDP
            rx = k
            sigx = np.sum(1-label[lis[:k]['index']])
            fdr = sigx / rx if rx > 0 else 0

            # GT FNR
            rx = size - k
            sigx = np.sum(label[lis[k:]['index']]) 
            fnr = sigx / rx if rx > 0 else 0

            # GT ATP
            atp = np.sum(label[lis[:k]['index']]) 
            return fdr, fnr, atp, signal_lis

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test_Statistics Generation')
    parser.add_argument('--device', default='cpu', type=str) # cpu, gpu
    parser.add_argument('--datapath', default='./', type=str)
    parser.add_argument('--labelpath', default='./', type=str)
    parser.add_argument('--savepath', default='./', type=str)
    args = parser.parse_args()  

    if torch.cuda.is_available() and args.device == 'gpu':
        args.device = 'cuda'

    x = np.load(args.datapath)
    model = nnHMRF_LIS(x, args.device, args.savepath)
    model.gem()
    gamma = np.load(os.path.join(args.savepath, 'gamma_shmrf.npy'))
    gt = np.load(args.labelpath).ravel()
    fdr, fnr, atp, signal_lis  = model.p_lis(gamma, threshold=0.1, label=gt)
