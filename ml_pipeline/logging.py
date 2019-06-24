class Logger(object):
    def __init__(self, logfilepath):
        self.logfilepath = logfilepath

    def writeToFile(self, txt):
        with open(self.logfilepath, 'a+') as logfile:
            logfile.write(txt + "\n")
            