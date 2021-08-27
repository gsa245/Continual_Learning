from scenarios import Experiment
from utils import readArgs

if __name__ == '__main__':

    parser = readArgs()
    args   = parser.parse_args()
    
    if True not in vars(args).values():
        print("No Valid Experiment argument passed")
    else:
        experiment = Experiment(args)
        experiment.run()
        experiment.plot()