"""WORK IN PROGRESS

Converting the functions into methods on a
DiffusionEntropy class to adhere to OoP and simplify
working with the methods.
"""
import time
from numpy import ndarray
from scipy import stats

class DiffusionEntropy:

    def __init__(self) -> object:
        self.data = ndarray(dtype=float)
        self.rounded_data = ndarray(dtype=float)
        self.events = ndarray(dtype=float)
        self.trajectory = ndarray(dtype=float)
        self.entropy = ndarray(dtype=float)
        self.window_lengths = ndarray(dtype=float)
        self.window_length_slice = ndarray(dtype=float)
        self.scaling = ndarray(dtype=float)
        self.intercept = ndarray(dtype=float)
        self.mu = ndarray(dtype=float)
    
    def sample_data(self, length: int, seed: float = time.time) -> object:
        import numpy as np
        """Generates an array of sample data.

        Parameters
        ----------
        self : object
            A DiffusionEntropy object.
        length : int
            The amount of sample data to generate.
        seed : Optional[float = time.time]
            Seed for random number generation. By default
            the system unix time is used.
        """
        np.random.seed(seed)
        random_steps = np.random.choice([-1, 1], length)
        random_steps[0] = 0  # Always start from 0
        random_walk = np.cumsum(random_steps)
        self.data = random_walk
    
    def apply_stripes(
            self,
            data: ndarray[float],
            stripes: int,
            show_data_plot: bool = False
        ) -> object:
        """
        Rounds `data` to `stripes` evenly spaced intervals.

        Parameters
        ----------
        self : object
            A DiffusionEntropy object.
        data : array_like
            Time-series data to be examined.
        stripes : int
            Number of stripes to apply. 
        show_data_plot : bool
            If True, show data plot with overlaid stripes.

        Returns
        ----------
        rounded_data : ndarray
            `data` rounded to `stripes` number of equally spaced intervals.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        if stripes < 2:
            raise ValueError("Parameter 'stripes' must be greater than 2.")
        if show_data_plot == True:
            lines = np.linspace(min(self.data), max(self.data), num=stripes)
            plt.figure(figsize=(5, 4))
            plt.plot(self.data)
            plt.hlines(
                y=lines,
                xmin=0,
                xmax=len(self.data),
                colors='0.3',
                linewidths=1,
                alpha=0.4
            )
            plt.xlabel('t')
            plt.ylabel('Data(t)')
            plt.title('Data with stripes')
            sns.despine()
            plt.show()
        if min(self.data) <= 0:
            self.data = self.data + abs(min(self.data))
        elif min(self.data) > 0:
            self.data = self.data - abs(min(self.data))
        max_data = max(self.data)
        min_data = min(self.data)
        data_width = abs(max_data - min_data)
        stripe_size = data_width / stripes
        rounded_data = self.data / stripe_size
        self.rounded_data = rounded_data
    
    def find_events(self) -> object:
        """Records an event (1) when `self.data` changes value.
        
        Parameters
        ----------
        self : object
            A DiffusionEntropy object.
        """
        import numpy as np
        events = []
        for i in range(1, len(self.data)):
            if (self.data[i] < np.floor(self.data[i-1])+1 and 
                self.data[i] > np.ceil(self.data[i-1])-1):
                # If both true, no crossing
                events.append(0)
            else:
                events.append(1)
        np.append(events, 0)
        setattr(self, "events", events)
    
    def make_trajectory(self) -> object:
        """Constructs diffusion trajectory from events."""
        import numpy as np
        trajectory = np.cumsum(self.events)
        self.trajectory = trajectory
    
    def get_entropy(self, trajectory: ndarray[float]) -> object:
        """
        Calculates the Shannon Entropy of the diffusion trajectory.

        Generates a range of window lengths L. Steps each one along 
        'trajectory' and computes the displacement of 'trajectory' 
        over each window position. Bins these displacements, and divides 
        by the sum of all bins to make the probability distribution 'p'. 
        Puts 'p' into the equation for Shannon Entropy to get s(L).
        Repeats for all L in range 'window_lengths'.

        Parameters
        ----------
        trajectory : array_like
            Diffusion trajectory. Constructed by make_trajectory.

        Returns
        ----------
        self : object
            A DiffusionEntropy object.
        s : ndarray
            Shannon Entropy values, S(L).
        window_lengths : ndarray
            Window lengths, L. 

        Notes
        ----------
        'tqdm(...)' makes the progress bar appear.
        """
        import numpy as np
        from tqdm import tqdm
        S = []
        window_lengths = np.arange(1, int(0.25*len(trajectory)), 1)
        for L in tqdm(window_lengths):
            window_starts = np.arange(0, len(trajectory)-L, 1)
            window_ends = np.arange(L, len(trajectory), 1)
            displacements = trajectory[window_ends] - trajectory[window_starts]
            counts, bin_edge = np.histogram(displacements, bins='doane')
            counts = np.array(counts[counts != 0])
            binsize = bin_edge[1] - bin_edge[0]
            P = counts / sum(counts)
            S.append(-sum(P*np.log(P)) + np.log(binsize))
        self.entropy = S
        self.window_lengths = window_lengths

    def get_no_stripe_entropy(self, trajectory: ndarray[float]) -> object:
        """
        Calculates the Shannon Entropy of the diffusion trajectory.

        Generates a range of window lengths L. Steps each one along 
        'trajectory' and computes the displacement of 'trajectory' 
        over each window position. Bins these displacements, and divides 
        by the sum of all bins to make the probability distribution 'p'. 
        Puts 'p' into the equation for Shannon Entropy to get s(L).
        Repeats for all L in range 'window_lengths'.

        Parameters
        ----------
        self : object
            A DiffusionEntropy object.
        trajectory : array_like
            Diffusion trajectory. FOR NO STRIPES JUST PASS THE DATA SERIES.

        Returns
        ----------
        S : ndarray
            Shannon Entropy values, S(L).
        window_lengths : ndarray
            Window lengths, L.

        Notes
        ----------
        `tqdm()` makes the progress bar appear.
        """
        import numpy as np
        from tqdm import tqdm
        window_lengths = np.arange(1, int(0.25*len(trajectory)), 1)
        S = []
        for L in tqdm(window_lengths):
            window_starts = np.arange(0, len(trajectory)-L, 1)
            window_ends = np.arange(L, len(trajectory), 1)
            traj = trajectory[window_starts] - trajectory[window_ends]
            counts, bin_edge = np.histogram(traj, bins='doane')  # Doane least bad for nongaussian
            counts = np.array(counts[counts != 0])
            binsize = bin_edge[1] - bin_edge[0]
            P = counts / sum(counts)
            S.append(-sum(P*np.log(P)) + np.log(binsize))
        self.entropy.append(S)
        self.window_lengths.append(window_lengths)
        self.entropy = S
        self.window_lengths = window_lengths

    def get_scaling(
            self,
            s: ndarray[float],
            L: ndarray[float],
            start: ndarray[float],
            stop: int,
            fit_method: str = "siegel"
        ) -> object:
        """
        Calculates scaling.
        
        Calculates the scaling of the time-series by performing a 
        least-squares linear fit over S(l) and ln(l).

        Parameters
        ----------
        self : object
            A DiffusionEntropy object.
        s : array_like
            Shannon Entropy values. 
        L : array_like
            Window Lengths. 
        start : int
            Index at which to start the fit slice.
        stop : int
            Index at which to stop the fit slice.
        fit_method : str {"siegel", "theilsen", "ls"}, optional
            Linear fit method to use. By default "siegel"

        Returns
        -------
        fit_slice_L : ndarray 
            The slice of window lengths L.
        coefficients : ndarray
            Slope and intercept of the fit. 

        Notes
        -----
        Prefer the siegel or theilsen methods. Least squares linear
        fits can introduce bias when done over log-scale data, see
        Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law
        distributions in empirical data. SIAM review, 51(4), pp.661-703.
        https://doi.org/10.1137/070710111.
        https://arxiv.org/pdf/0706.1062.pdf.
        """
        import numpy as np
        if fit_method not in ["siegel", "theilsen", "ls"]:
            raise ValueError(
                "Parameter 'method' must be one of: \
                    ['siegel', 'theilsen', 'ls']."
            )
        s_slice = s[start:stop]
        L_slice = L[start:stop]
        if fit_method == "ls":
            coefficients = np.polyfit(np.log(L_slice), s_slice, 1)
        if fit_method == "theilsen":
            coefficients = stats.theilslopes(s_slice, np.log(L_slice))
        if fit_method == "siegel":
            
            coefficients = stats.siegelslopes(s_slice, np.log(L_slice))
        self.window_length_slice = L_slice
        self.scaling = coefficients[0]
        self.intercept = coefficients[1]

    def get_mu(self) -> object:
        """
        Calculates the mu.

        Parameters
        ----------
        scaling : float
            Scaling of the time-series process. 

        Returns
        ----------
        mu : float
            Complexity parameter. Powerlaw index for inter-event 
            time distribution.
        Notes
        ----------
        mu is calculated by both rules. later both are plotted
        against the line relating delta and mu, to hopefully
        let users graphically determine the correct mu.
        """
        mu1 = 1 + self.scaling
        mu2 = 1 + (1 / self.scaling)
        self.mu.append(mu1)
        self.mu.append(mu2)
        
    def plot_mu_candidates(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Initialize plotting vars
        x1 = np.linspace(1, 2, 100)
        x2 = np.linspace(2, 3, 100)
        x3 = np.linspace(3, 4, 100)
        y1 = x1 - 1
        y2 = 1 / (x2 - 1)
        y3 = np.full(100, 0.5)
        # Make plot
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x1, y1, color='k')
        ax.plot(x2, y2, color='k')
        ax.plot(x3, y3, color='k')
        ax.plot(
            self.mu[0],
            self.scaling,
            marker='o',
            label=f'$\\mu$ = {np.round(self.mu[0], 2)}'
        )
        ax.plot(
            self.mu[1],
            self.scaling,
            marker='o',
            label=f'$\\mu$ = {np.round(self.mu[1], 2)}'
        )
        ax.set_xticks(ticks=np.linspace(1, 4, 7))
        ax.set_yticks(ticks=np.linspace(0, 1, 5))
        ax.set_xlabel('$\\mu$')
        ax.set_ylabel('$\\delta$')
        ax.legend(loc=0)
        ax.grid(True)
        sns.despine(left=True, bottom=True)
        plt.show(fig)
    
    def dea_no_stripes(
            self,
            data: ndarray[float],
            start: int,
            stop: int,
            fit_method: str = "siegel"
        ) -> object:
        """
        Applies DEA without the stripes refinement.

        Original DEA. Takes the original time series as the diffusion 
        trajectory.

        Parameters
        ----------
        data : array_like
            Time-series to be analysed.
        start : int
            Array index at which to start linear fit.
        stop : int 
            Array index at which to stop linear fit.
        fit_method : str {"siegel", "theilsen", "ls"}, optional
            Linear fit method to use. By default "siegel"

        Returns
        ----------
        figure 
            A figure plotting S(l) vs. ln(l), overlaid with the fit 
            line, labelled with the scaling and mu values.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("Beginning DEA without stripes.")
        S, L = self.get_no_stripe_entropy(data)
        fit = self.get_scaling(S, L, start, stop, fit_method)
        mu = self.get_mu(fit[1][0])

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(L, S, linestyle='', marker='.', alpha=0.5)
        ax.plot(
            fit[0],
            fit[1][0] * np.log(fit[0]) + fit[1][1],
            color='k',
            label=f'$\delta = {np.round(fit[1][0], 3)}$'
        )
        ax.plot(
            [], [], linestyle='', label=f'$\mu = {np.round(mu, 3)}$')
        ax.xscale('log')
        ax.xlabel('$ln(l)$')
        ax.ylabel('$S(l)$')
        ax.legend(loc=0)
        ax.grid(False)
        ax.tick_params(
            which="major",
            bottom=True,
            left=True,
            length=5,
            color="#cccccc"
        )
        ax.tick_params(
            which="minor",
            bottom=True,
            left=True,
            length=3,
            color="#cccccc"
        )
        sns.despine(trim=True)
        plt.show(fig)
        print("DEA without stripes complete.")
        return self
    
    def dea_with_stripes(
            self,
            data: ndarray[float],
            stripes: int,
            start: int,
            stop: int,
            show_data_plot: bool = False,
            fit_method: str = "siegel"
        ) -> object:
        """
        Applies DEA with the stripes refinement.

        Runs a sequence of functions to apply stripes and then 
        perform DEA on the data series. 

        Parameters
        ----------
        data : array_like
            Time-series to be analysed.
        stripes : int
            Number of stripes to be applied to the data.
        start : int
            Array index at which to start linear fit.
        stop : int 
            Array index at which to stop linear fit.
        show_data_plot : bool
            If True, show data plot with overlaid stripes.
        fit_method : str {"siegel", "theilsen", "ls"}, optional
            Linear fit method to use. By default "siegel"

        Returns
        ----------
        fig : figure 
            A figure plotting S(l) vs. ln(l), overlaid with the fit 
            line, labelled with the scaling and mu values.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("Beginning DEA with stripes.")
        rounded_data = self.apply_stripes(data, stripes, show_data_plot)
        event_array = self.find_events(rounded_data)
        diffusion_trajectory = self.make_trajectory(event_array)
        s, L = self.get_entropy(diffusion_trajectory)
        fit = self.get_scaling(s, L, start, stop, fit_method)
        mu = self.get_mu(self.scaling)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(L, s, linestyle='', marker='.', alpha=0.5)
        ax.plot(
            self.window_lengths,
            self.scaling * np.log(self.window_lengths) + self.intercept,
            color='k',
            label=f'$\delta = {np.round(self.scaling, 3)}$'
        )
        ax.set_xscale('log')
        ax.set_xlabel('$ln(L)$')
        ax.set_ylabel('$S(L)$')
        ax.legend(loc=0)
        ax.grid(False)
        ax.tick_params(
            which="major",
            bottom=True,
            left=True,
            length=5,
            color="#cccccc"
        )
        ax.tick_params(
            which="minor",
            bottom=True,
            left=True,
            length=3,
            color="#cccccc"
        )
        sns.despine(trim=True)
        plt.show(fig)

        self.plot_mu_candidates(fit[1][0], self.mu[0], self.mu[1])
        print("DEA with stripes complete.")
        return self
