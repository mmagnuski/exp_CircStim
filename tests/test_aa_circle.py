from psychopy import visual, event

window = visual.Window(monitor='testMonitor', blendMode='add', useFBO=True)
circle = visual.Circle(window, edges=16, radius=0.5, interpolate=True)
circle.setFillColor((-0.1, -0.1, -0.1))
circle.setLineColor((-0.2, -0.2, -0.2))

circle.draw()
# window.getMovieFrame('back')
window.flip()
window.getMovieFrame()

circle.radius = 0.7
circle.draw()
window.flip()
window.getMovieFrame()

circle.radius = 0.3
circle.draw()
window.flip()
window.getMovieFrame()

window.saveMovieFrames('test_aa.png')
event.waitKeys()

