# various viz functions
from __future__ import absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt

from .utils import check_color, group

def plot_weibull(weibull, x=None, pth='', ax=None, points=True, line=True,
			 	 mean_points=False, min_bucket='adaptive',
				 split_bucket='adaptive', line_color=None, contrast_steps=None,
				 mean_points_color=(0.22, 0.58, 0.78), linewidth=3.):
	# set up x:
	if x is None:
		x = weibull.x

	# get predicted data
	numpnts = 1000
	lowest = -40. if weibull.kind == 'weibull_db' else 0.
	highest = 2. if weibull.kind == 'weibull_db' else 1.2
	vmin, vmax = max(lowest, x.min()), min(highest, x.max())
	data_range = vmax - vmin
	vmin, vmax = (max(lowest, vmin - 0.1 * data_range),
				  min(vmax + 0.1 * data_range, highest))
	x = np.linspace(vmin, vmax, num=numpnts)
	y = weibull.predict(x)

	# add noise to y data to increase visibility
	l = len(x)
	yrnd = np.random.uniform(-0.065, 0.065, l)

	if line_color is None:
	    line_color = 'seaborn_green'

	# plot setup
	if ax is None:
	    f, ax = plt.subplots()
	try:
		ax.set_facecolor((0.92, 0.92, 0.92))
	except:
		ax.set_axis_bgcolor((0.92, 0.92, 0.92))
	plt.grid(True, color=(1., 1., 1.), lw=1.5, linestyle='-', zorder=-1)

	line_color = check_color(line_color)

	# plot buckets
	# TODO - bucketing or binning points should be done by sep fun
	if mean_points:
	    from scipy import stats
	    mean_points_color = check_color(mean_points_color)

	    # bucketize
	    # ---------
	    # get points and sort
	    x_pnts = x.copy()
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
	    plt.scatter(x, weibull.orig_y + yrnd, alpha=0.6, lw=0,
	        zorder=6, c=[0.3, 0.3, 0.3])
	if line:
	    plt.plot(x, y, zorder=5, lw=linewidth, color=line_color)

	# aesthetics
	# ----------
	uplim = min(vmax, np.where(y == y.max())[0][0])
	lowlim = vmin
	plt.xlim([lowlim, uplim])
	plt.ylim([-0.1, 1.1])
	plt.xlabel('stimulus intensity')
	plt.ylabel('correctness')

	# save figure
	if pth:
	    tempfname = os.path.join(pth, 'weibull_fit_temp.png')
	    plt.savefig(tempfname, dpi=120)
	    plt.close()
	    return tempfname


def plot_quest_plus(qp):
	import matplotlib.gridspec as gridspec
	from .weibull import weibull_db, Weibull

	# check cmap
	cmaps = dir(plt.cm)
	use_cmap = 'viridis' if 'viridis' in cmaps else 'jet'

	# create figure and axes overlay
	fig = plt.figure(figsize=(15, 6))
	gs = gridspec.GridSpec(3, 3)
	im_ax = plt.subplot(gs[:, 0])
	prob_ax1 = plt.subplot(gs[0, 1])
	prob_ax2 = plt.subplot(gs[1, 1])
	prob_ax3 = plt.subplot(gs[2, 1])
	entropy_ax = plt.subplot(gs[0, -1])
	func_ax = plt.subplot(gs[1:, -1])

	# get posterior
	posterior = qp.get_posterior().copy()

	# plot posterior agregated along lapse dimension
	model_threshold, model_slope, model_lapse = qp._orig_params
	thresh_step = np.diff(model_threshold).mean()
	im_ax.imshow(posterior.sum(axis=-1), aspect='auto',
				 cmap=use_cmap, interpolation='none',
				 extent=[-0.5, len(model_slope) + 0.5,
				 		 model_threshold[-1] + thresh_step / 2.,
						 model_threshold[0] - thresh_step / 2.])
	x_ind = np.arange(0, len(model_slope), 3)
	im_ax.set_xticks(x_ind)
	im_ax.set_xticklabels(['{:.2f}'.format(x) for x in model_slope[x_ind]])
	im_ax.set_xlabel('slope')
	im_ax.set_ylabel('threshold')

	# aggregated parameter probabilities
	axes = [prob_ax1, prob_ax2, prob_ax3]
	xs = [model_threshold, model_slope, model_lapse]
	reduce_dims = [(1, 2), (0, 2), (0, 1)]
	titles = ['threshold', 'slope', 'lapse']
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
	for i in range(3):
		# for _ in range(i):
		#     axes[i]._get_lines.get_next_color()
		this_prob = posterior.sum(axis=reduce_dims[i])
		axes[i].grid(True)
		if i == 1:
			this_x = np.arange(len(xs[i]))
			axes[i].plot(this_x, this_prob, color=colors[i])
			axes[i].fill_between(this_x, this_prob, color=colors[i], alpha=0.5)
			ticks = this_x[::3]
			axes[i].set_xticks(ticks)
			axes[i].set_xticklabels(['{:.2f}'.format(x)
									 for x in xs[i][ticks]])
		else:
			axes[i].plot(xs[i], this_prob, color=colors[i])
			axes[i].fill_between(xs[i], this_prob, color=colors[i], alpha=0.5)
		axes[i].set_title('{} probability'.format(titles[i]))

	# plot expected entropy
	entropy_ax.plot(qp.stim_domain, qp.entropy, color='k')
	entropy_ax.set_title('expected entropy for next contrast')

	# get model parameters
	current_params = qp.param_domain[posterior.argmax(), :]
	mean_params = (qp.posterior[:, np.newaxis] * qp.param_domain).sum(axis=0)

	# psychometric function fit
	w = Weibull(kind='weibull_db')
	w.fit(np.array(qp.stim_history), np.array(qp.resp_history), current_params)
	w.plot(ax=func_ax, linewidth=1.5)
	func_ax.findobj(plt.Line2D)[0].set_label('Maximum Likelihood fit')

	vmin, vmax = qp.stim_domain[[0, -1]]
	x = np.linspace(vmin, vmax, num=1000)
	func_ax.plot(x, qp.function(x, current_params),
				 label='Bayesian max prob fit', zorder=10)
	func_ax.plot(x, qp.function(x, mean_params), label='Bayesian mean prob fit',
				 zorder=11)
	func_ax.legend(loc='best')
	fig.tight_layout()
	return fig
