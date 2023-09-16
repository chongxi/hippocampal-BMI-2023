import sklearn
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d
import os
import matplotlib.pyplot as plt

def load_pretrained_model(rat='rat1', task='Jumper'):
    '''
    Load a pre-trained model.

    Parameters:
    --------------------
    rat : str
        Name of the rat. (e.g., 'rat1', 'rat2', 'rat3')
    task : str
        Name of the task. (e.g., 'Jumper', 'Jedi')

    Returns:
    --------------------
    model : object
        A trained model.
    neuron_idx : array
        Indices of neurons to be used for decoding. (place units)
    B_bins : int
        B_bins time-bins of PV are fed into the deepnet at a rate of 0.1 seconds per bin (i.e., 50 bins=5s, 15 bins=1.5s)
    '''
    # load Jumper models
    if rat == 'rat1' and task == 'Jumper':
        B_bins = 50
        neuron_idx = np.load('./data/Jumper/wr112_0905_place_units_id.npy')
        model = load_model('./pretrained_model/wr112_0905_Jumper_model_state_dict.pth', neuron_idx=neuron_idx, B_bins=B_bins)
    
    elif rat == 'rat2' and task == 'Jumper':
        B_bins = 50
        neuron_idx = np.load('./data/Jumper/wr118_0821_place_units_id.npy')
        model = load_model('./pretrained_model/wr118_0821_Jumper_model_state_dict.pth', neuron_idx=neuron_idx, B_bins=B_bins)

    elif rat == 'rat3' and task == 'Jumper':
        B_bins = 15 # this data used a 1.5 s window for real-time deepnet decoding
        neuron_idx = np.load('./data/Jumper/wr121_0927_place_units_id.npy')
        model = load_model('./pretrained_model/wr121_0927_Jumper_model_state_dict.pth', neuron_idx=neuron_idx, B_bins=B_bins)

    # load Jedi models
    if rat == 'rat1' and task == 'Jedi':
        B_bins = 50
        neuron_idx = np.load('./data/Jedi/wr112_0901_place_units_id.npy')
        model = load_model('./pretrained_model/wr112_0901_Jedi_model_state_dict.pth', neuron_idx=neuron_idx, B_bins=B_bins)
    
    if rat == 'rat2' and task == 'Jedi':
        B_bins = 50
        neuron_idx = np.load('./data/Jedi/wr118_0829_place_units_id.npy')
        model = load_model('./pretrained_model/wr118_0829_Jedi_model_state_dict.pth', neuron_idx=neuron_idx, B_bins=B_bins)     

    if rat == 'rat3' and task == 'Jedi':
        B_bins = 50
        neuron_idx = np.load('./data/Jedi/wr121_0915_place_units_id.npy')
        model = load_model('./pretrained_model/wr121_0915_Jedi_model_state_dict.pth', neuron_idx=neuron_idx, B_bins=B_bins)           

    return model, neuron_idx, B_bins

def load_model(model_path, neuron_idx, B_bins):
    '''
    Load a pre-trained model.

    Parameters
    ----------
    model_path : str
        Path to the model file.

    neuron_idx : array
        Indices of neurons to be used for decoding. (place units)
    
    B_bins : int
        Number of time bins for each neuron to input to the model.

    Returns
    -------
    model : object
        A trained model.
    '''
    decoder = DeepOSC(input_dim=neuron_idx.shape[0]*B_bins, hidden_dim=[256, 256], output_dim=2)
    decoder.model.load_state_dict(torch.load(model_path))

    return decoder.model


def spike_noise_bernoulli(X, noise_level=1, p=0.5, gain=1, cuda=True, IID=True):
    '''
    Add IID noise to data (spike vector count) to train network to ignore off-manifold activity

    each neuron will add a noise to each time bin: 
    noise = uniform(-noise_level, noise_level) * bernoulli(p)
    '''
    if cuda:
        noise = torch.ones_like(X).uniform_(-noise_level, noise_level) * \
            torch.bernoulli(torch.ones_like(X)*p).cuda()
        X = X.cuda()
    else:
        noise = torch.ones_like(
            X).uniform_(-noise_level, noise_level)*torch.bernoulli(torch.ones_like(X)*p)
    if IID:
        X = torch.relu(gain*(X + noise))
    else:
        X = torch.relu(gain*(X + noise*X.mean(axis=0)))
    return X


def spike_noise_gaussian(X, noise_level=1, mean=0.0, std=3.0, gain=1, cuda=True, IID=True):
    '''
    Add IID noise to data (spike vector count) to train network to ignore off-manifold activity

    each neuron will add a noise to each time bin: 
    noise = uniform(-noise_level, noise_level) * bernoulli(p)
    '''
    if cuda:
        noise = torch.ones_like(X).uniform_(-noise_level, noise_level) * \
            torch.normal(torch.ones_like(X)*mean, std).cuda()
        X = X.cuda()
    else:
        noise = torch.ones_like(
            X).uniform_(-noise_level, noise_level)*torch.normal(torch.ones_like(X)*mean, std)
    if IID:
        X = torch.relu(gain*(X + noise))
    else:
        X = torch.relu(gain*(X + noise*X.mean(axis=0)))
    return X


class FA(FactorAnalysis):
    '''
    Factor analysis extended from sklearn
    `reconstruct` input to 
        1. remove independent noise (i.e., off-manifold activity) across neurons
        2. keep only the shared variability (on-manifold activity) across neurons
        The result is not integer.
    `sample_manifold` input to 
        acquire resampled on-manifold spike count (integer)

    Parameters
        X: array-like of shape (n_samples, n_features)

    Example
        fa = FA(n_components=30)
        fa.fit(scv) #scv (spike count vector): (n_samples, n_units)
        reconstructed_scv = fa.reconstruct(scv) 
        resampled_scv = fa.sample_manifold(scv) 
    '''

    def reconstruct(self, X):
        factors = self.transform(X)
        reconstructed_X = np.clip(
            factors@self.components_ + self.mean_, 0, 100)
        return reconstructed_X

    def sample_manifold(self, X, sampling_method=np.random.poisson):
        reconstructed_X = self.reconstruct(X)
        sampled_X = sampling_method(reconstructed_X)
        return sampled_X


class Decoder(object):
    """Base class for the decoders for place prediction"""
    def __init__(self, t_window, t_step=None, verbose=True):
        '''
        t_window is the bin_size
        t_step   is the step_size (if None then use pc.ts as natrual sliding window)
        https://github.com/chongxi/spiketag/issues/47 
        
        For Non-RNN decoder, large bin size in a single bin are required
        For RNN decoder,   small bin size but multiple bins are required

        During certain neural state, such as MUA burst (ripple), a small step size is required 
        (e.g. t_window:20ms, t_step:5ms is used by Pfeiffer and Foster 2013 for trajectory events) 

        dec.partition(training_range, valid_range, testing_range, low_speed_cutoff) 
        serves the cross-validation
        https://github.com/chongxi/spiketag/issues/50
        '''
        self.t_window = t_window
        self.t_step   = t_step
        self.verbose  = verbose

    @property
    def B_bins(self):
        self._b_bins = int(np.round(self.t_window / self.t_step))
        return self._b_bins

    def connect_to(self, pc):
        '''
        This decoder is specialized for position decoding
        Connect to a place-cells object that contains behavior, neural data and co-analysis
        '''
        # self.pc = pc
        self.pc = copy.deepcopy(pc)
        self.pc.rank_fields('spatial_bit_spike') # rerank the field
        self.fields = self.pc.fields
        if self.t_step is not None:
            print('Link the decoder with the place cell object (pc):\r\n resample the pc according to current decoder input sampling rate {0:.4f} Hz'.format(1/self.t_step))
            self.pc(t_step=self.t_step)

    def drop_neuron(self, _disable_neuron_idx):
        if type(_disable_neuron_idx) == int:
            _disable_neuron_idx = [_disable_neuron_idx]
        self._disable_neuron_idx = _disable_neuron_idx
        if self._disable_neuron_idx is not None:
            self.neuron_idx = np.array([_ for _ in range(self.fields.shape[0]) if _ not in self._disable_neuron_idx])

    def resample(self, t_step=None, t_window=None):
        if t_window is None:
            t_window = self.binner.bin_size*self.binner.B
        elif t_window != self.t_window:
            self.t_window = t_window
        if t_step is None:
            t_step = self.binner.bin_size
        elif t_step != self.t_step:
            self.t_step   = t_step
            self.connect_to(self.pc)

    def _percent_to_time(self, percent):
        len_frame = len(self.pc.ts)
        totime = int(np.round((percent * len_frame)))
        if totime < 0: 
            totime = 0
        elif totime > len_frame - 1:
            totime = len_frame - 1
        return totime

    def partition(self, training_range=[0.0, 0.5], valid_range=[0.5, 0.6], testing_range=[0.5, 1.0],
                        low_speed_cutoff={'training': True, 'testing': False}, v_cutoff=None):

        self.train_range = training_range
        self.valid_range = valid_range
        self.test_range  = testing_range
        self.low_speed_cutoff = low_speed_cutoff

        if v_cutoff is None:
            self.v_cutoff = self.pc.v_cutoff
        else:
            self.v_cutoff = v_cutoff

        self.train_time = [self.pc.ts[self._percent_to_time(training_range[0])], 
                           self.pc.ts[self._percent_to_time(training_range[1])]]
        self.valid_time = [self.pc.ts[self._percent_to_time(valid_range[0])], 
                           self.pc.ts[self._percent_to_time(valid_range[1])]]
        self.test_time  = [self.pc.ts[self._percent_to_time(testing_range[0])], 
                           self.pc.ts[self._percent_to_time(testing_range[1])]]

        self.train_idx = np.arange(self._percent_to_time(training_range[0]),
                                   self._percent_to_time(training_range[1]))
        self.valid_idx = np.arange(self._percent_to_time(valid_range[0]),
                                   self._percent_to_time(valid_range[1]))
        self.test_idx  = np.arange(self._percent_to_time(testing_range[0]),
                                   self._percent_to_time(testing_range[1]))

        if low_speed_cutoff['training'] is True:
            self.train_idx = self.train_idx[self.pc.v_smoothed[self.train_idx]>self.v_cutoff]
            self.valid_idx = self.valid_idx[self.pc.v_smoothed[self.valid_idx]>self.v_cutoff]

        if low_speed_cutoff['testing'] is True:
            self.test_idx = self.test_idx[self.pc.v_smoothed[self.test_idx]>self.v_cutoff]

        if self.verbose:
            print('{0} training samples\n{1} validation samples\n{2} testing samples'.format(self.train_idx.shape[0],
                                                                                             self.valid_idx.shape[0],
                                                                                             self.test_idx.shape[0]))

    def save(self, filename):
        torch.save(self, filename)

    def get_data(self, minimum_spikes=2, remove_first_unit=False):
        '''
        Connect to pc first and then set the partition parameter. After these two we can get data
        The data strucutre is different for RNN and non-RNN decoder
        Therefore each decoder subclass has its own get_partitioned_data method
        In low_speed periods, data should be removed from train and valid:
        '''
        assert(self.pc.ts.shape[0] == self.pc.pos.shape[0])

        self.pc.get_scv(self.t_window); # t_step is None unless specified, using pc.ts
        self.pc.output_variables = ['scv', 'pos']
        X, y = self.pc[:]
        assert(X.shape[0]==y.shape[0])

        self.train_X, self.train_y = X[self.train_idx], y[self.train_idx]
        self.valid_X, self.valid_y = X[self.valid_idx], y[self.valid_idx]
        self.test_X,  self.test_y  = X[self.test_idx], y[self.test_idx]

        if minimum_spikes>0:
            self.train_X, self.train_y = mua_count_cut_off(self.train_X, self.train_y, minimum_spikes)
            self.valid_X, self.valid_y = mua_count_cut_off(self.valid_X, self.valid_y, minimum_spikes)
            self.test_X,  self.test_y  = mua_count_cut_off(self.test_X,  self.test_y,  minimum_spikes)

        if remove_first_unit:
            self.train_X = self.train_X[:,1:]
            self.valid_X = self.valid_X[:,1:]
            self.test_X  = self.test_X[:,1:]

        return (self.train_X, self.train_y), (self.valid_X, self.valid_y), (self.test_X, self.test_y) 

    def r2_score(self, y_true, y_predict, multioutput=True):
        '''
        use sklearn.metrics.r2_score(y_true, y_pred, multioutput=True)
        Note: r2_score is not symmetric, r2(y_true, y_pred) != r2(y_pred, y_true)
        '''
        if multioutput is True:
            score = r2_score(y_true, y_predict, multioutput='raw_values')
        else:
            score = r2_score(y_true, y_predict)
        if self.verbose:
            print('r2 score: {}\n'.format(score))
        return score

    def auto_pipeline(self, t_smooth=2, remove_first_unit=False, firing_rate_modulation=True):
        '''
        example for evaluate the funciton of acc[partition]:
        >>> dec = NaiveBayes(t_window=500e-3, t_step=60e-3)
        >>> dec.connect_to(pc)
        >>> r_scores = []
        >>> partition_range = np.arange(0.1, 1, 0.05)
        >>> for i in partition_range:
        >>>     dec.partition(training_range=[0, i], valid_range=[0.5, 0.6], testing_range=[i, 1],
        >>>                   low_speed_cutoff={'training': True, 'testing': True})
        >>>     r_scores.append(dec.auto_pipeline(2))
        '''
        (X_train, y_train), (X_valid, y_valid), (self.X_test, self.y_test) = self.get_data(minimum_spikes=2, 
                                                                                           remove_first_unit=remove_first_unit)
        self.fit(X_train, y_train, remove_first_unit=remove_first_unit)
        self.predicted_y = self.predict(self.X_test, 
                                        firing_rate_modulation=firing_rate_modulation, 
                                        two_steps=False)
        self.smooth_factor  = int(t_smooth/self.pc.t_step) # 2 second by default
        # self.sm_predicted_y = smooth(self.predicted_y, self.smooth_factor)
        self.sm_predicted_y = gaussian_filter1d(self.predicted_y, sigma=10, axis=0) * 1.1
        score = self.r2_score(self.y_test, self.sm_predicted_y) # ! r2 score is not symmetric, needs to be (true, prediction)
        return score

    def score(self, t_smooth=2, remove_first_unit=False, firing_rate_modulation=True):
        '''
        dec.score will first automatically train the decoder (fit) and then test it (predict). 
        The training set and test set are also automatically saved in dec.X_train and dec.X_test
        The training and test label are saved in dec.y_train and dec.y_test
        '''
        return self.auto_pipeline(t_smooth=t_smooth, 
                                  remove_first_unit=remove_first_unit,
                                  firing_rate_modulation=firing_rate_modulation)

    def plot_decoding_err(self, real_pos, dec_pos, err_percentile=90, N=None, err_max=None):
        err = abs(real_pos - dec_pos)
        dt = self.t_step
        if N is None:
            N = err.shape[0]
        return plot_err_2d(real_pos, dec_pos, err, dt, N, err_percentile, err_max)


class Olayer(nn.Module):
    '''
    O(Oscillatory)layer was inspired by siren paper https://arxiv.org/abs/2006.09661 using sinusoidal activation function.
    (x,y) position is the input for siren, here we reverse the input-output, thus (x,y) is output for us using stacked Olayers for internal processing.

    Change Olayer to linear layer may lead to two specific effects: 
        1. Output can occationally go out of the maze, which is annoying for online real live experiment. That never happens for Olayer
            - This ponentially is related to "Sirens can be leveraged to solve challenging boundary value problems" in siren paper.
        2. Olayer trains more stable than linear layer, test score keep going up and the final R2 score in test set is higher. 
            - This is our empirical observation. 
    '''
    def __init__(self, hidden_dim=[128, 64]):
        super(Olayer, self).__init__()
        self.fc1l = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc1r = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc1m = nn.Linear(hidden_dim[0], hidden_dim[1])

    def forward(self, xp, freq=torch.pi):
        xl = self.fc1l(xp)      # hidden_dim[0] -> hidden_dim[1]
        xr = self.fc1r(xp)
#         xm = self.fc1m(xp)
        xg = torch.cos(freq*xl) + torch.cos(freq*xr)  # Olayer
        # xg = xl + xr # Linear
        return xg


class SineDec(nn.Module):

    def __init__(self, input_dim, hidden_dim=[128, 64], output_dim=2, bn=False, LSTM=True):
        super(SineDec, self).__init__()
        self.LSTM = LSTM
        self.bn = bn
        self.encoder = nn.Linear(input_dim, hidden_dim[0])
        self.ln1 = nn.LayerNorm(hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])
        # self.bn1 = nn.InstanceNorm1d(hidden_dim[0])
        self.fc1 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc1xm = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc2g = nn.Linear(hidden_dim[1], output_dim, bias=True)
        self.fc2p = nn.Linear(hidden_dim[1], output_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim[1], output_dim, bias=True)
        # self.olayer1 = Olayer(hidden_dim)
        # self.olayer2 = Olayer(hidden_dim)
        # self.olayer3 = Olayer(hidden_dim)
        # self.olayer4 = Olayer(hidden_dim)
        # self.olayer5 = Olayer(hidden_dim)
        # self.olayer6 = Olayer(hidden_dim)
        # self.olayer7 = Olayer(hidden_dim)
        # self.olayer8 = Olayer(hidden_dim)
        self.olayer = nn.ModuleList([Olayer(hidden_dim) for i in range(11)])
        self.dropout = nn.Dropout(0.1)
        # alternative (not used yet)
        # input: (batch, seq, feature)
        self.lstm = nn.LSTM(hidden_dim[0], hidden_dim[0], batch_first=True)
        self.ln_lstm = nn.LayerNorm(hidden_dim[0])
        # speed related
        self.fcx2v = nn.Linear(hidden_dim[1], output_dim, bias=True)
        self.v2x_lstm = nn.LSTM(output_dim, hidden_dim[1], bias=True)
        self.ln2 = nn.LayerNorm(hidden_dim[1])
        self.fcv2x = nn.Linear(output_dim, hidden_dim[1], bias=True)
        self.scale = 1

    def forward(self, X):
        # x = self.dropout(X)
        x = self.encoder(X)  # input_dim -> hidden_dim[0]
        # x = self.ln1(x)
        if self.bn:
            x = self.bn1(x)      #

        # x = self.dropout(x)
#         x = torch.sin(x)
        # x = self.fc1(x)    # to both place and speed prediction
        x = F.softmax(F.relu(self.fc1(x))) * x
        # x = F.softmax(F.relu(self.fc1(x))) * x

        if self.LSTM:
            x, h = self.lstm(x.view(len(x), 1, -1))
            x = self.ln_lstm(x)
            x.squeeze_()

        xg = self.olayer[0](x) + x
        xv = self.olayer[1](x) + x

        xv = self.olayer[2](xv) + xv
#         xv = F.relu(xv)
#         xv = torch.sin(xv)/(xv+1e-15)
        # xv = self.dropout(xv)
        v = self.fcx2v(xv)
#         v2x, _ = self.v2x_lstm(v.view(len(v), 1, -1))
#         v2x = self.ln2(v2x)
#         v2x.squeeze_()
#         xg = self.dropout(v2x) # heavy dropout for grid emergence
#         xg = v2x
        xg = self.olayer[3](xg) + xg
        xg = self.olayer[4](xg) + xg
        xg = self.olayer[5](xg) + xg
        # xg = self.olayer[6](xg) + xg
        # xg = self.olayer[7](xg) + xg
        # xg = self.olayer[8](xg) + xg
        # xg = self.olayer[9](xg) + xg
        # xg = self.olayer[10](xg) + xg
        # xg = self.dropout(xg)
        # prediction
        y = self.fc2p(xg)      # hidden_dim[1] -> 2

#         xg = self.olayer7(xg)
#         xg = self.olayer8(xg)

#         xp = self.fc2p(xp)
        return x, xg, y, v

    def predict(self, X, cuda=True, mode='train', bn_momentum=0.1, target='pos', smooth=False):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X).float()
        if X.ndim == 1:
            X = X.view(1, -1)
        if cuda:
            X = X.cuda()
        if mode == 'eval':
            self.eval()
        elif mode == 'train':
            self.train()
        self.bn1.momentum = bn_momentum

        with torch.inference_mode():
            _, _, _yo, _vo = self.forward(X)
            _yo = _yo.cpu().detach().numpy()

        if target == 'pos':
            if smooth:
                # multiply by 1.1 to counter the decreasing amplitude effect due to the gaussian filter
                _yo = gaussian_filter1d(_yo, sigma=10, axis=0) * 1.1
            return np.nan_to_num(_yo)
        if target == 'motion':
            _mo = F.sigmoid(_vo.norm(dim=1)).cpu(
            ).detach().numpy().reshape(-1, 1)
            return np.nan_to_num(_mo)
        if target == 'vel':
            _vo = _vo.cpu().detach().numpy()
            return np.nan_to_num(_vo)

    def predict_rt(self, X, neuron_idx, cuda, mode, bn_momentum):
        '''
        T_steps can be 1
        X: (T_steps, B_bins, N_neurons)
        y: (T_steps, N_neurons)

        Note: we trained on square root of spike count, so we need to sqrt it back when predicting in real time
        '''
        X = X[..., neuron_idx]
        X = X.ravel()
        X = np.sqrt(X)
        y = self.predict(X, cuda, mode, bn_momentum)
        return y


class DeepOSC(Decoder):
    """
    DeepOSC Decoder for position prediction (input X, output y) 
    where X is the spike bin matrix (T_step, N_neurons*B_bins) # ! this is different from naive bayes decoder 
    where y is the 2D position (x,y)

    Examples:
    -------------------------------------------------------------
    from spiketag.analysis import DeepOSC, smooth

    dec = DeepOSC(t_window=250e-3, t_step=50e-3)
    dec.connect_to(pc)

    dec.partition(training_range=[0.0, .7], valid_range=[0.5, 0.6], testing_range=[0.6, 1.0], 
                  low_speed_cutoff={'training': True, 'testing': True})
    (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = dec.get_data(minimum_spikes=0)
    dec.fit(train_X, train_y)

    predicted_y = dec.predict(test_X)
    dec_pos  = smooth(predicted_y, 60)
    real_pos = test_y
    
    score = dec.r2_score(real_pos, dec_pos)

    # optional (cost time):
    # dec.plot_decoding_err(real_pos, dec_pos);

    To get scv matrix to hack (check the size):
    -------------------------------------------------------------
    # ! 1. data
    pc.output_variables = ['scv', 'pos']
    N = int(len(pc)*0.5)
    X,y = pc[:N]
    X_test, y_test = pc[N:]
    # ! 2. training
    dec.train(X,y) 
    # ! 3. predict   
    _y = dec.predict(X_test)
    y_decoded  = smooth(_y, 60)

    # ! 4. test
    score = dec.r2_score(y_test, y_decoded)

    Test real-time prediction:
    -------------------------------------------------------------
    _y = dec.predict_rt()
    """

    # TODO
    pass

    def __init__(self, input_dim, hidden_dim=[256, 256], output_dim=2, bn=True, LSTM=True, t_window=5, t_step=0.1):
        super(DeepOSC, self).__init__(t_window, t_step)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = SineDec(input_dim=self.input_dim,
                             hidden_dim=self.hidden_dim,
                             output_dim=self.output_dim,
                             bn=bn,
                             LSTM=LSTM)
        self.test_r2 = []
        self.losses = []
        self.running_steps = 0
        self._running_data = []
        self.update_interval = int(60/t_step)  # update bn every 60 seconds

    @property
    def running_data(self):
        return np.vstack(self._running_data)

    def unroll(self, scv, n):
        '''
        unroll scv such that it has (T_step, B_bins, N_neurons) structure
        '''
        pass

    def fit(self, X, y, X_test, y_test, max_epoch=5000, smooth_factor=30, max_noise=1,
            early_stop_r2=0.82, lr=3e-4, weight_decay=0.01, cuda=True,
            target='pos'):
        '''
        training the deep neural network, using GPU if `cuda` == True
        '''
        # optmizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        if type(X) == np.ndarray:
            X = torch.from_numpy(X).float()
        if type(y) == np.ndarray:
            y = torch.from_numpy(y).float()
        if type(X_test) == np.ndarray:
            X_test = torch.from_numpy(X_test).float()
        if type(y_test) == np.ndarray:
            y_test = torch.from_numpy(y_test).float()

        if cuda:
            X = X.cuda()
            y = y.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
            self.model.cuda()

        for epoch in range(max_epoch):
            self.model.train()
            self.optimizer.zero_grad()
            gain = 1 + 0.05*torch.randn(1, device='cuda')
            std = 0.00 + np.abs(np.sin(epoch/600*3.14)) * max_noise
            # gausian noise with sqrt spike count
            # noise_X = spike_noise_gaussian(X, noise_level=1,
            #                               mean=0, std=std, gain=1,
            #                               IID=True, cuda=True)
            # bernoulli noise with spike count
            noise_X = spike_noise_bernoulli(
                X, noise_level=std, p=0.5, gain=1, cuda=True, IID=True)
            h, grid, _y, _v = self.model(noise_X)
            # regression for pos: y should be (N, 2)
            if target == 'pos' or target == 'vel':
                # now_location = y # (y + 0.2*torch.rand_like(y, device='cuda'))
                loss = F.mse_loss(y, _y)
            # binary classification for motion: y should be (N,), each value is either 0 or 1
            elif target == 'motion':
                move_or_not = _v.norm(dim=1)
                loss = F.binary_cross_entropy_with_logits(move_or_not, y)
        #         loss = F.mse_loss(y[:-1] + _v[:-1], y[1:]) * 5
        #     loss += F.mse_loss(y[:-1] + _v[:-1], y[1:])
        #     loss += F.mse_loss(_v, v)
        #     loss_g = F.mse_loss(now_location, h)
            norm_val = torch.norm(grid, p=1, dim=1).sum() * 1e-6 + \
                torch.norm(grid, p=1, dim=0).sum() * 1e-6
            norm_h = torch.norm(h, p=1) * 1e-6
            # loss = loss + norm_val  # norm_h # + norm_val # + loss_g # + norm_val # + norm_h
        #     grid = nn.Dropout(0.6)(grid)
        #     grid_pos = self.model.fc2(grid)
        #     grid_pos_loss = F.mse_loss(grid_pos, now_location)
        #     loss += grid_pos_loss

            loss.backward()
            self.optimizer.step()
        #     lrtim.step()
            if epoch > 1000:
                self.set_learning_rate(lr/2)
            if epoch > 2000:
                self.set_learning_rate(lr/4)

            if epoch % 10 == 0:
                with torch.inference_mode():
                    _yo = self.predict(X_test, target=target)
                # dec_y = smooth(_yo, smooth_factor)
                dec_y = gaussian_filter1d(_yo, sigma=10, axis=0) * 1.1
                r2_p = r2_score(y_test.cpu().numpy(), dec_y)
                self.test_r2.append(r2_p)
                self.losses.append(loss.item())
                print(f'[{epoch}] loss: {loss.item():.3f}, test r2: {r2_p:.3f}, std:{std:.3f}, norm_h:{norm_h:.3f}, norm_output:{norm_val:.3f}', end='\r')

    def set_learning_rate(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def plot_loss(self):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = 'C0'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('test R2', color=color)
        ax1.plot(np.arange(len(self.test_r2))*10, self.test_r2, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'C1'
        # we already handled the x-label with ax1
        ax2.set_ylabel('training losses', color=color)
        ax2.plot(np.arange(len(self.losses))*10, self.losses, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        return fig

    def predict(self, X, cuda=True, mode='eval', bn_momentum=0.1, target='pos'):
        y = self.model.predict(X, cuda=cuda, mode=mode,
                               bn_momentum=bn_momentum, target=target)
        return y

    def disable_running_stats(self):
        '''
        during training we don't track the running mean and var
        '''
        self.model.bn1.track_running_stats = False
        self.model.bn1.running_mean = None
        self.model.bn1.running_var = None

    def enable_running_stats(self, cuda=True, reset=True):
        '''
        during online inference mode we track the running mean and var in bn layer
        we do this because firing rate of neuron could suffer from covariate shift
        Track the running mean and var can hopefully compensate the neuron drifting effect
        '''
        self.model.bn1.track_running_stats = True
        if reset: # reset the running mean and var to zeros (only do this at the beginning of the experiment)
            if cuda:
                self.model.bn1.running_mean = torch.zeros((256,)).float().cuda()
                self.model.bn1.running_var = torch.zeros((256,)).float().cuda()
            else:
                self.model.bn1.running_mean = torch.zeros((256,)).float()
                self.model.bn1.running_var = torch.zeros((256,)).float()

    def update_bn(self, cuda=True, bn_momentum=0.9):
        self.predict(self.running_data, cuda=cuda,
                     mode='train', bn_momentum=bn_momentum)

    def predict_rt(self, X, cuda=True, mode='eval', bn_momentum=0.1):
        # predict in real time eval mode
        # note: we don't need to take the squre root here, as the `model.predict_rt` will take the square root
        y = self.model.predict_rt(
            X, neuron_idx=self.neuron_idx, cuda=cuda, mode=mode, bn_momentum=bn_momentum)

        # cache data for computing running mean and std
        self.running_steps += 1
        self._running_data.append(X[..., self.neuron_idx].ravel())

        # update running mean and std to BN (batch normalization)
        if self.running_steps % self.update_interval == 0 and self.running_steps > 600:
            self.update_bn(cuda=cuda)
        return y
