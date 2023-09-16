import numpy as np
from scipy.ndimage import gaussian_filter1d

def get_sliding_pv(pv, n):
    _scv_list = [pv[i:i-n] if i < n else pv[n:] for i in range(n+1)]
    new_scv = np.hstack(_scv_list)
    return new_scv

def pv_2_spv(pv, B_bins=50, padding=False):
    '''
    convert population vector to sliding population vector
    (N_samples, N_neurons) ---> (N_samples, B_bins, N_neurons)

    Parameters
    ----------
    pv : array: (n_samples, n_neurons)
        Population vector.
    B_bins : int
        Number of bins for sliding population vector.

    Returns
    -------
    X : array: (n_samples, B_bins, n_neurons)
        Sliding population vector.
    X[i] is the a snapshot of the last B_bins of population vector, X[i][-1] is the current population vector.
    '''
    pv[:, 0] = 0
    X = get_sliding_pv(pv, B_bins-1)
    if padding: # pad B_bins-1 of zeros bins to the beginning of X
        X = np.vstack([np.zeros((B_bins-1, X.shape[1])), X])
    X = X.reshape(-1, B_bins, pv.shape[-1])
    return X

def update_bn(model, running_data, cuda=True):
    model.predict(running_data, cuda=cuda, mode='train', bn_momentum=0.9)

def decode(model, X, neuron_idx, cuda=True, mimic_realtime=True, smooth=False):
    '''
    Decode CA1 neural activity to position using a trained model.

    Parameters
    ----------
    model : object
        A trained model.
    X : array: (n_samples, n_time_steps, n_neurons)
        Neural activity data.
    neuron_idx : array
        Indices of neurons to be used for decoding. (place units)
    cuda : bool
        Whether to use GPU.
    mimic_realtime : bool
        Whether to mimic real-time decoding. If True, the bn layer of the model will be updated every minute. 

        - mimic_realtime=True: frame-by-frame decoding (exact offline decoding, except for first 2 minutes)
            As this is a time-consuming process (it decode every 100 ms and update the bn layer every minute), it can take a few minutes to finish.
            After 2 minutes of bn update, the output should be the same as the real-time decoding (<1cm mean decoding deviation from the real-time result in a 100 by 100cm maze). 

        - mimic_realtime=False: batch decoding (approximate offline decoding)
            if mimic_realtime is False, the output will not be the same as the real-time decoding. But it will be much faster (using cuda) and 
            the output should be still close to the offline decoding (~10cm mean decoding deviation from the real-time result in a 100cm by 100cm maze).
    '''

    if cuda:
        model.cuda();

    if mimic_realtime:
        redec_pos = []
        running_data = []
        update_interval = 600
        bn_momentum = 0.1
        running_steps = 0

        for _bmi_scv in X:
            # offline BMI prediction
            xy = model.predict_rt(_bmi_scv, neuron_idx=neuron_idx, cuda=cuda, mode='eval', bn_momentum=bn_momentum)
            # update running data
            running_steps += 1
            running_data.append(_bmi_scv[..., neuron_idx].ravel())
            # update BN (batch normalization) layer every 600 samples (1 minute)
            if running_steps % update_interval == 0 and running_steps > update_interval:
                update_bn(model, np.vstack(running_data), cuda)
            redec_pos.append(xy)

        redec_pos = np.array(redec_pos).squeeze()
    
    else:
        # batch BMI prediction, the input dimension is n_samples * (n_time_steps * n_place_units)
        redec_pos = model.predict(X[:,:,neuron_idx].reshape(X.shape[0], -1), cuda=cuda, mode='train')
    
    if smooth:
        redec_pos = gaussian_filter1d(redec_pos, sigma=10, axis=0)

    return redec_pos
