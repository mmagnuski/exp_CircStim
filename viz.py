# various viz functions
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import check_color, group

def plot_weibull(weibull, pth='', ax=None, points=True, line=True,
			 	 mean_points=False, min_bucket='adaptive',
				 split_bucket='adaptive', line_color=None, contrast_steps=None,
				 mean_points_color=(0.22, 0.58, 0.78)):
	# get predicted data
	numpnts = 1000
	x = np.linspace(0., 2., num=numpnts)
	y = weibull.predict(x)

	# add noise to y data to increase visibility
	l = len(weibull.x)
	yrnd = np.random.uniform(-0.065, 0.065, l)

	if line_color is None:
	    line_color = 'seaborn_red'

	# plot setup
	if ax is None:
	    f, ax = plt.subplots()
	ax.set_axis_bgcolor((0.92, 0.92, 0.92))
	plt.grid(True, color=(1., 1., 1.), lw=1.5, linestyle='-', zorder=-1)

	line_color = check_color(line_color)

	# plot buckets
	if mean_points:
	    from scipy import stats
	    mean_points_color = check_color(mean_points_color)

	    # bucketize
	    # ---------
	    # get points and sort
	    x_pnts = weibull.x.copy()
	    y_pnts = weibull.orig_y.copy()
	    srt = x_pnts.argsort()
	    x_pnts = x_pnts[srt]
	    y_pnts = y_pnts[srt]

	    # adaptive min_bucket and split_bucket
	    adaptive_min_bucket = isinstance(min_bucket, str) and \
	        min_bucket == 'adaptive'
	    adaptive_split_bucket = isinstance(split_bucket, str) and \
	        split_bucket == 'adaptive'
	    if adaptive_min_bucket or adaptive_split_bucket:
	        drop_elements = int(np.ceil(len(x_pnts) * 0.08))
	        contrast_range = x_pnts[drop_elements:-drop_elements][[0, -1]]
	        contrast_range = np.diff(contrast_range)[0]
	        if adaptive_min_bucket:
	            min_bucket = contrast_range / 20.
	        if adaptive_split_bucket:
	            split_bucket = contrast_range / 10.

	    # look for buckets
	    x_buckets = group(np.diff(x_pnts) <= min_bucket)
	    n_pnts_in_bucket = (np.diff(x_buckets, axis=-1) + 1).ravel()
	    good_buckets = n_pnts_in_bucket >= (3 - 1) # -1 because of diff

	    if x_buckets.shape[0] > 0 and np.any(good_buckets):
	        x_buckets = x_buckets[good_buckets, :]

	        # turn buckets to slices, get mean and sem
	        x_buckets[:, 1] += 2 # +1 because of python slicing
								 # another +1 because of diff
	        slices = [slice(l[0], l[1]) for l in x_buckets]

	        # test each slice for contrast range and split if needed
	        add_slices = list()
	        ii = 0

	        while ii < len(slices):
	            slc = slices[ii]
	            pnts = x_pnts[slc]
	            l, h = pnts[[0, -1]]
	            rng = h - l
	            start = slc.start
	            if rng > split_bucket:
	                slices.pop(ii)
	                n_full_splits = int(np.floor(rng / split_bucket))
	                last_l = l
	                last_ind = 0
	                for splt in range(n_full_splits):
	                    this_high = last_l + split_bucket
	                    this_ind = np.where(pnts >= this_high)[0][0]
	                    add_slices.append(slice(start + last_ind, start +
												this_ind + 1))
	                    last_l = this_high
	                    last_ind = this_ind + 1

	                # last one - to the end
	                if start + last_ind < slc.stop:
	                    add_slices.append(slice(start + last_ind, slc.stop))
	            else:
	                ii += 1
	        slices.extend(add_slices)

	        x_bucket_mean = np.array([x_pnts[slc].mean()
	                                  for slc in slices])
	        y_bucket_mean = np.array([y_pnts[slc].mean()
	                                  for slc in slices])
	        bucket_sem = np.array([stats.sem(y_pnts[slc])
	                               for slc in slices])

	        # plot bucket means and sem
	        plt.scatter(x_bucket_mean, y_bucket_mean, lw=0, zorder=4, s=32.,
	                    c=mean_points_color)
	        plt.vlines(x_bucket_mean,
	                   y_bucket_mean - bucket_sem,
	                   y_bucket_mean + bucket_sem,
	                   lw=2, zorder=4, colors=mean_points_color)

	if contrast_steps is not None:
	    corrs = weibull.predict(contrast_steps)
	    plt.vlines(contrast_steps, corrs - 0.04, corrs + 0.04,
	               lw=2, zorder=4, colors=[0., 0., 0.])

	if points:
	    plt.scatter(weibull.x, weibull.orig_y + yrnd, alpha=0.6, lw=0,
	        zorder=6, c=[0.3, 0.3, 0.3])
	if line:
	    plt.plot(x, y, zorder=5, lw=3, color=line_color)

	# aesthetics
	# ----------
	gab99 = weibull.get_threshold([0.95])[0]
	if gab99 < 0. or gab99 > 2.:
	    maxval = 2.
	else:
	    maxval = gab99 + 0.1
	if weibull.params[0] <= 0:
	    maxval = np.max(weibull.x) + 0.1
	uplim = min([2.0, np.round(maxval, decimals=1)])
	plt.xlim([0.0, uplim])
	plt.ylim([-0.1, 1.1])
	plt.xlabel('stimulus intensity')
	plt.ylabel('correctness')

	# save figure
	if pth:
	    tempfname = os.path.join(pth, 'weibull_fit_temp.png')
	    plt.savefig(tempfname, dpi=120)
	    plt.close()
	    return tempfname
