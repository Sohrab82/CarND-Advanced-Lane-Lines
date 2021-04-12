from matplotlib.pyplot import figtext


class Lane():
    def __init__(self):
        self.fitx = None
        self.fit_coeff = None
        self.use_histo = True
        self.conf = 0
        self.ploty = None

    def update(self, fitx, fit_coeff, ploty, conf):
        self.fitx = fitx
        self.fit_coeff = fit_coeff
        self.ploty = ploty
        self.conf = conf
