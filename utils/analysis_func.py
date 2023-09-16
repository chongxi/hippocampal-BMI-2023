import numpy as np
import pandas as pd
import scipy.stats as stats


def time_2_trial_no(t, trial_start, trial_end):
    '''
    Convert time to trial number.
    trial_start: an array of trial start times
    trial_end: an array of trial end times
    return index in trial_start and trial_end that 
    trial_start[index] <= t <= trial_end[index]

    return: trial_no is an array of length 0 or 1
    len(trial_no) == 0 if t is not in any trial
    len(trial_no) == 1 if t is in one trial
    '''
    trial_no = np.where((trial_start <= t) & (t <= trial_end))[0]
    return trial_no


def convert_frame_to_time(i, B_bins=50):
    '''
    Convert frame index to time in seconds.
    B_bins: number of bins in a sliding window of population vector
    - rat1: 50
    - rat2: 50
    - rat3: 15
    '''
    time = (i+B_bins-1)*0.1
    return time


def trajectory_2_angle(trajectory, goal=None):
    """
    Converts a trajectory to an angle. If a goal is provided, the angle is relative to the goal.

    Parameters:
    trajectory (array): Array of trajectory.
    goal (array): Array of the goal location. If None, the angle is relative to the previous point.

    Returns:
    array: Array of angles.
    """
    if goal is None:
        hd_vec = np.diff(trajectory, axis=0)
        angles = np.arctan2(hd_vec[:, 1], hd_vec[:, 0])
        return hd_vec, angles
    else:
        hd_vec = np.diff(trajectory, axis=0)
        angles = np.arctan2(hd_vec[:, 1], hd_vec[:, 0])
        goal_vec = goal - trajectory[:-1]
        goal_angles = np.arctan2(goal_vec[:, 1], goal_vec[:, 0])
        relative_angles = angles - goal_angles
        return hd_vec, relative_angles


def subsample_20(arr, num_subsamples=20):
    # Compute the number of elements to include in each subsample
    if len(arr) < num_subsamples:
        return arr
    subsample_size = len(arr) // num_subsamples
    # Use numpy.arange() with a step size of 2 to subsample the array
    subsamples = np.arange(0, len(arr), subsample_size)
    if len(subsamples) > num_subsamples:
        subsamples = subsamples[-num_subsamples:]
    return arr[subsamples]


def get_total_angles(trials, rat_name='wr112', subsample=20):
    total_angles = []
    for i in range(len(trials[rat_name])):
        _goal_pos = trials[rat_name][i]['goal_pos']
        _bmi_t, _bmi_pos = trials[rat_name][i]['bmi_t'], trials[rat_name][i]['bmi_pos']
        # subsample in total 20 points from _bmi_pos
        # _bmi_pos_subsampled = _bmi_pos[::_bmi_pos.shape[0]//subsample]
        _bmi_pos_subsampled = subsample_20(_bmi_pos, num_subsamples=20)
        # _bmi_pos_subsampled = _bmi_pos[::2]
        head_directions, angles = trajectory_2_angle(_bmi_pos_subsampled, _goal_pos)
        angles = angles % (2*np.pi)
        # print(angles.shape)
        total_angles.append(angles)
    total_angles = np.hstack(total_angles)
    return total_angles


class TimeSeries(object):
    '''
    Time series with different sampling rate, different start offset, and different length can cause headache when 
    analyzing and visualizing them. This class is designed to extract and align time series without resampling. 

    sig = TS(t, data, name='sig')
    sig_ROI = sig.between(0.5, 1.5) # return another TS object with t and data between 0.5 and 1.5
    sig_ROI.plot(ax=ax) # plot the ROI

    Examples:
    ---------
    # 1. load data from different sources (with different sampling rate)
    lfp = TS(t = lfp_ch18['t'], data = lfp_ch18['lfp'], name='lfp')
    unit = UNIT(bin_len=0.1, nbins=50) # 100ms bin, 50 bins
    unit.load_unitpacket('./fet.bin')
    bmi_time = unit.bin_index/10
    hdv = np.fromfile('./animal_hdv.bin', np.float32).reshape(-1,2)
    hd, v = hdv[:,0] - 90, hdv[:,1]

    # 2. load lfp,spk,vel as a TimeSeries object
    lfp = TS(t = lfp_18['t'], data = lfp_18['lfp'], name='lfp')
    spk = TS(t = unit.spike_time[unit.spike_id!=0], data = unit.spike_id[unit.spike_id!=0], name='spike_timing')
    vel = TS(t = bmi_time, data=v, name='ball_velocity')

    # 3. extract common ROI
    t_start, t_end = 737.6, 738.6
    _lfp = lfp.between(t_start, t_end)
    _spk = spk.between(t_start, t_end)
    _bv  = vel.between(t_start, t_end)

    ### check the ROI time points, they can be very different: _lfp.t.shape, _bv.t.shape, _spk.t.shape

    # 4. plot together although they have different length in the same time period
    fig, ax = plt.subplots(3,1,figsize=(15,8), sharex=True)
    _spk.scatter(ax = ax[0], c=_spk.data, s=2, cmap='Set2')
    _lfp.plot(ax = ax[1]);
    _bv.plot(ax = ax[2]);
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    fig.tight_layout()
    ---------
    '''

    def __init__(self, t=None, data=None, name=None):
        self.t = t if t is not None else np.arange(data.shape[0])
        self.data = data if data is not None else np.zeros_like(self.t)
        self.name = name if name is not None else ''
        # self.fs = 1/(t[1]-t[0])
        self.ndim = self.data.ndim
        if self.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        self.nch = self.data.shape[1]
        assert(len(self.t) == self.data.shape[0]
               ), 'time and data length do not match'

    def select(self, feature_idx):
        return TimeSeries(self.t, self.data[:, feature_idx], self.name)

    def between(self, start_time, end_time):
        idx = np.where((self.t >= start_time) & (self.t <= end_time))[0]
        return TimeSeries(self.t[idx], self.data[idx], self.name)

    def exclude(self, start_time, end_time):
        idx = np.where((self.t < start_time) | (self.t > end_time))[0]
        return TimeSeries(self.t[idx], self.data[idx], self.name)

    def searchsorted(self, ts, side='left'):
        # subsample from current timeseries according to `ts`
        # return a new timeseries object with the same length as ts, in which the data
        # occurs at self.t that is closest to ts
        #
        # ts can be a subset of self.t, and the returned index can be used to
        # extract the corresponding data from self.data that happens at the closest time to ts
        # That being said, self.t and ts do not have to be the same length, but
        # for each time in ts, there should be a corresponding time in self.t
        # usually self.t has higher sampling rate than ts
        # then we can use searchsorted to specifically subsample according to ts
        idx = np.searchsorted(self.t, ts, side=side)
        return TimeSeries(self.t[idx], self.data[idx])

    def mean(self, axis=1):
        return TimeSeries(self.t, self.data.mean(axis=axis), self.name+'_mean')

    def std(self, axis=1):
        return TimeSeries(self.t, self.data.std(axis=axis), self.name+'_std')

    def sum(self, axis=1):
        return TimeSeries(self.t, self.data.sum(axis=axis), self.name+'_sum')

    def diff(self, axis=0):
        '''
        diff in time, not in feature (axis=0)
        diff in feature, not in time (axis=1)
        '''
        return TimeSeries(self.t[1:], np.diff(self.data, axis=axis), self.name+'_diff')

    def norm(self, axis=1, ord=None):
        '''
        norm in feature, not in time (axis=1)
        norm in time, not in feature (axis=0)
        '''
        return TimeSeries(self.t, np.linalg.norm(self.data, axis=axis, ord=ord), self.name+'_norm')

    def min_subtract(self):
        return TimeSeries(self.t, self.data - np.min(self.data, axis=0), self.name+'_mean_subtract')

    def max_subtract(self):
        return TimeSeries(self.t, self.data - np.max(self.data, axis=0), self.name+'_mean_subtract')

    def mean_subtract(self):
        return TimeSeries(self.t, self.data - np.mean(self.data, axis=0), self.name+'_mean_subtract')

    def interp1d(self, dt=10e-3):
        from scipy.interpolate import interp1d
        assert(self.t.shape[0] == self.data.ravel().shape[0])
        f = interp1d(self.t, self.data.ravel(), fill_value="interpolate")
        new_t = np.arange(self.t[0], self.t[-1], dt)
        new_data = f(new_t)
        return TimeSeries(new_t, new_data.reshape(-1, 1), self.name+'_interp1d')

    def moving_sum_1d(self, window_size=10, axis=0):
        # use np.convolve to calculate moving average, remove .ravel will cause error
        new_data = np.convolve(
            self.data.ravel(), np.ones(window_size), mode='same')
        return TimeSeries(self.t, new_data, self.name+'_moving_sum')

    def ci(self, alpha=0.95, func=stats.t):
        '''
        calculate the confidence interval of the data, by default use t distribution
        can alsue us func=stats.norm for normal distribution
        '''
        # by default use t distribution, t distribution is good when sample size is small
        # but when sample size is large, t distribution is close to normal distribution
        ci = func.interval(alpha=alpha, df=len(self.data)-1,
                           loc=np.mean(self.data),
                           scale=stats.sem(self.data))
        return ci

    def find_peaks(self, high=None, low=None, beta_std=None, **kwargs):
        """
        This function identifies peak segments in a signal by identifying local maxima that exceed a specified height threshold. 
        The height threshold is calculated as the mean of the signal plus a multiple of the standard deviation. 
            - high_treshold = mean+beta_std*std
            - height = high_treshold if beta_std is None
        The start and end indices of each peak segment are then determined by finding the first point on either side of the peak 
        that falls below a second threshold, which is calculated as the mean minus a multiple of the standard deviation.
            - low_threshold = mean-beta_std*std
            - low_threshold = mean-std if beta_std is None

        This function iterates over channels and returns dctionary where the keys are the channel id. 
        
        Parameters:
        - beta_std (float, optional): A parameter that determines the peak segments. , 
        - **kwargs: Additional keyword arguments to pass to the `signal.find_peaks()` function.
        
        Returns:
        - peaks (dict): A dictionary where the keys are the channel indices and the values are TimeSeries objects containing the data at the peak indices.
        - left (dict): A dictionary where the keys are the channel indices and the values are TimeSeries objects containing the data at the start indices of the segments.
        - right (dict): A dictionary where the keys are the channel indices and the values are TimeSeries objects containing the data at the end indices of the segments.
        """
        peaks, left, right = {}, {}, {}
        for ch in range(self.nch):
            if beta_std is not None:
                self.high_threshold = np.mean(
                    self.data[:, ch]) + beta_std * np.std(self.data[:, ch])
                self.low_threshold = np.mean(
                    self.data[:, ch]) - beta_std * np.std(self.data[:, ch])
                _peaks_idx, _ = signal.find_peaks(
                    self.data[:, ch], height=self.high_threshold, **kwargs)
                _left_idx, _right_idx = self.find_left_right_nearest(
                    np.where(self.data[:, ch] < self.low_threshold)[0], _peaks_idx)
            elif high is not None and low is not None:
                _peaks_idx, _ = signal.find_peaks(
                    self.data[:, ch], height=high, **kwargs)
                _left_idx, _right_idx = self.find_left_right_nearest(
                    np.where(self.data[:, ch] < low)[0], _peaks_idx)
            peaks[ch] = TimeSeries(
                self.t[_peaks_idx], self.data[_peaks_idx], self.name+'_peaks_'+str(ch))
            left[ch] = TimeSeries(
                self.t[_left_idx], self.data[_left_idx], self.name+'_left_'+str(ch))
            right[ch] = TimeSeries(
                self.t[_right_idx], self.data[_right_idx], self.name+'_right_'+str(ch))
        return peaks, left, right

    def find_left_right_nearest(self, x_idx, v_idx):
        """
        Find the adjacent index of v_idx (N,) in x_idx (return the N left index of a, and N right index of a)
        """
        assert(len(x_idx) > 1), 'x_idx must contains more than one element'
        _idx_right = np.searchsorted(x_idx, v_idx)
        _idx_left = np.searchsorted(x_idx, v_idx) - 1
        left = x_idx[_idx_left]  # - 1
        right = x_idx[_idx_right]
        return left, right

    def test_find_left_right_nearest(self):
        x_idx = np.array([1, 3, 5, 7, 9])
        v_idx = np.array([2, 4, 6])
        expected_left = np.array([1, 3, 5])
        expected_right = np.array([3, 5, 7])

        # Call the find_left_right_nearest method
        left, right = self.find_left_right_nearest(x_idx, v_idx)

        # Assert that the output is as expected
        np.testing.assert_array_equal(left, expected_left)
        np.testing.assert_array_equal(right, expected_right)

    def filtfilt(self, N=20, Wn=[100, 300], type='bp', fs=None, show=False):
        if fs is None:
            fs = 1/(self.t[1]-self.t[0])

        b, a = signal.butter(N, Wn, btype=type, fs=fs)
        y = signal.filtfilt(b, a, self.data, axis=0)

        if show is True:
            import matplotlib.pyplot as plt
            w, h = signal.freqz(b, a, fs=fs)
            plt.plot(w, 20 * np.log10(abs(h)))
            plt.axvspan(Wn[0], Wn[1], alpha=0.5)
            plt.title('Butterworth filter frequency response')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude [dB]')
        return TimeSeries(self.t, y, self.name+'_filtered'+str(Wn))

    def hilbert(self, **kwargs):
        '''
        self.data must be 1-d numpy array
        '''
        amplitude_envelope = np.abs(
            signal.hilbert(self.data.ravel(), **kwargs))
        return TimeSeries(self.t, amplitude_envelope, self.name+'_hilbert')

    def zscore(self, **kwargs):
        return TimeSeries(self.t, zscore(self.data, **kwargs), self.name+'_zscore')

    def smooth(self, n=5, type='gaussian'):
        '''
        - for gaussian, n is sigma
        - for boxcar,   n is window length
        '''
        if type == 'boxcar':
            data = smooth(self.data.astype(np.float32), n)
            return TimeSeries(self.t, data, self.name+f'_smooth_{n}')
        elif type == 'gaussian':
            data = gaussian_filter1d(self.data.astype(
                np.float32), sigma=n, axis=0, mode='constant')
            return TimeSeries(self.t, data, self.name+f'_gaussian_smooth_{n}')

    def get_cwt(self, fmin=0, fmax=128, dj=1/100, show=False):
        cwtmatr = get_cwt(self.t, self.data.ravel(),
                          fmin=fmin, fmax=fmax, dj=dj)
        if show is True:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
            self.plot(ax=ax[0])
            ax[1].pcolormesh(cwtmatr.t, cwtmatr.freq,
                             cwtmatr.magnitude, cmap='viridis')
        return cwtmatr

    def plot(self, ax=None, **kwargs):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(self.t, self.data, **kwargs)
        return ax

    def scatter(self, ax=None, **kwargs):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.scatter(self.t, self.data, **kwargs)
        return ax

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        return TimeSeries(self.t[idx], self.data[idx], self.name)

    def __len__(self):
        return len(self.t)

    def __repr__(self):
        return f'{self.name} from {self.t[0]} to {self.t[-1]}'

    def __add__(self, other):
        return TimeSeries(self.t, self.data + other.data, self.name)

    def __sub__(self, other):
        return TimeSeries(self.t, self.data - other.data, self.name)

    def __mul__(self, other):
        return TimeSeries(self.t, self.data * other.data, self.name)

    def __truediv__(self, other):
        return TimeSeries(self.t, self.data / other.data, self.name)

    def __pow__(self, other):
        return TimeSeries(self.t, self.data ** other.data, self.name)

    def __neg__(self):
        return TimeSeries(self.t, -self.data, self.name)

    def __abs__(self):
        return TimeSeries(self.t, np.abs(self.data), self.name)

    def __eq__(self, other):
        return TimeSeries(self.t, self.data == other.data, self.name)

    def __ne__(self, other):
        return TimeSeries(self.t, self.data != other.data, self.name)

    def __lt__(self, other):
        return TimeSeries(self.t, self.data < other.data, self.name)

    def __le__(self, other):
        return TimeSeries(self.t, self.data <= other.data, self.name)

    def __gt__(self, other):
        return TimeSeries(self.t, self.data > other.data, self.name)

    def __ge__(self, other):
        return TimeSeries(self.t, self.data >= other.data, self.name)

    def __and__(self, other):
        return TimeSeries(self.t, self.data & other.data, self.name)

    def __or__(self, other):
        return TimeSeries(self.t, self.data | other.data, self.name)

    def __xor__(self, other):
        return TimeSeries(self.t, self.data ^ other.data, self.name)

    def __invert__(self):
        return TimeSeries(self.t, ~self.data, self.name)

    def __lshift__(self, other):
        return TimeSeries(self.t, self.data << other.data, self.name)

    def __rshift__(self, other):
        return TimeSeries(self.t, self.data >> other.data, self.name)

    def __iadd__(self, other):
        self.data += other.data
        return self

    def __isub__(self, other):
        self.data -= other.data
        return self

    def __imul__(self, other):
        self.data *= other.data
        return self

    def __itruediv__(self, other):
        self.data /= other.data
        return self

    def __ipow__(self, other):
        self.data **= other.data
        return self

    def __iand__(self, other):
        self.data &= other.data
        return self

    def __ior__(self, other):
        self.data |= other.data
        return self

    def __ixor__(self, other):
        self.data ^= other.data
        return self

    def __ilshift__(self, other):
        self.data <<= other.data
        return self

    def __irshift__(self, other):
        self.data >>= other.data
        return self
