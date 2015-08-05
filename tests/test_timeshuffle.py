from stimutils import TimeShuffle

times = TimeShuffle(start=1.5, end=3., every=0.5,
			times=2, shuffle=False)
print times.all()