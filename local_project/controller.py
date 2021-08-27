from experiments import Experiments
from models import TestModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tl_vs_cl", action="store_true")
    parser.add_argument("--exp_sample_replacement", action="store_true")
    parser.add_argument("--exp_buffer_size", action="store_true")
    args = parser.parse_args()
    experiments = Experiments()

    if args.exp_tl_vs_cl:
        print("> Experiment: Continual Learning VS Transfer Learning")
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="CONTINUAL_VS_TRANSFER_LEARNING",
                                                    usecase="Continual Learning", replay_size=7500,
                                                    random_selection=True)
        experiments.runTransferLearningExperiment(experiment_name="CONTINUAL_VS_TRANSFER_LEARNING",
                                                  usecase="Transfer Learning")
        experiments.plotExperiment(experiment_name="CONTINUAL_VS_TRANSFER_LEARNING",
                                   title="Continual VS Transfer Learning (CORe50 NICv2 - 391)")
    elif args.exp_sample_replacement:
        print("> Experiment: FIFO VS Random Selection - Buffer Sample Replacement")
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="FIFO_VS_RANDOM_SELECTION_BS_10000", usecase="FIFO",
                                                    replay_size=10000, random_selection=False)
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="FIFO_VS_RANDOM_SELECTION_BS_10000",
                                                    usecase="RANDOM SELECTION", replay_size=10000,
                                                    random_selection=True)
        experiments.plotExperiment(experiment_name="FIFO_VS_RANDOM_SELECTION_BS_10000",
                                   title="FIFO VS Random Selection (CORe50 NICv2 - 391)")
    elif args.exp_buffer_size:
        print("> Experiment: Replay Buffer Size")
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="REPLAY_BUFFER_SIZE_EXPERIMENTS",
                                                    usecase="RBS_3000",
                                                    replay_size=3100, random_selection=True)
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="REPLAY_BUFFER_SIZE_EXPERIMENTS",
                                                    usecase="RBS_5000",
                                                    replay_size=5000, random_selection=True)
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="REPLAY_BUFFER_SIZE_EXPERIMENTS",
                                                    usecase="RBS_7500",
                                                    replay_size=7500, random_selection=True)
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="REPLAY_BUFFER_SIZE_EXPERIMENTS",
                                                    usecase="RBS_10000",
                                                    replay_size=10000, random_selection=True)
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="REPLAY_BUFFER_SIZE_EXPERIMENTS",
                                                    usecase="RBS_15000",
                                                    replay_size=15000, random_selection=True)
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="REPLAY_BUFFER_SIZE_EXPERIMENTS",
                                                    usecase="RBS_20000",
                                                    replay_size=20000, random_selection=True)
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="REPLAY_BUFFER_SIZE_EXPERIMENTS",
                                                    usecase="RBS_30000",
                                                    replay_size=30000, random_selection=True)
        experiments.plotExperiment(experiment_name="REPLAY_BUFFER_SIZE_EXPERIMENTS",
                                   title="Replay Buffer Size Experiments (CORe50 NICv2 - 391)")
    else:
        print("> No valid experiment option provided")