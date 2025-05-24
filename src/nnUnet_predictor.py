import subprocess
import os
import pathlib as pl
import time

class Resegmentor():
    def __init__(self, multimodal_set=None, singlemodal_set=None, combined_set = None):
        self.multimodal_set = multimodal_set
        self.singlemodal_set = singlemodal_set
        self.combined_set = combined_set

    def execute(self, task=['524', '504']):
        env = os.environ.copy()
        env["RESULTS_FOLDER"] = "/home/lorenz/BMPipeline/resegmentation"
        time.sleep(1)
        if '524' in task:
            if os.listdir(self.multimodal_set):
                command_multi = [
                "nnUNet_predict",
                "-i", self.multimodal_set,
                "-o", self.multimodal_set.parent/(self.multimodal_set.name+'_predictions'),
                '-tr', 'nnUNetTrainerV2_Loss_DiceCE_noSmooth',
                '-ctr', 'nnUNetTrainerV2CascadeFullRes',
                '-m', '3d_fullres',
                '-p', 'nnUNetPlansv2.1',
                '-t', 'Task524_BrainMetsResegMultimod1to3'
                ]
                # Run the command
                print(f'== running multimodal prediction on source data {self.multimodal_set}')
                subprocess.run(command_multi, env=env)
                print(f'''== saved multimodal prediction on source data {self.multimodal_set.parent/(self.multimodal_set.name+'_predictions')}''')
            else:
                print('== skipping multimodal prediction, found no files in directory')


        if '504' in task:
            if os.listdir(self.singlemodal_set):
                command_single = [
                "nnUNet_predict",
                "-i", self.singlemodal_set,
                "-o", self.singlemodal_set.parent/(self.singlemodal_set.name+'_predictions'),
                '-tr', 'nnUNetTrainerV2_Loss_DiceCE_noSmooth',
                '-ctr', 'nnUNetTrainerV2CascadeFullRes',
                '-m', '3d_fullres',
                '-p', 'nnUNetPlansv2.1',
                '-t', 'Task504_BrainMetsReseg1to3'
                ]
                # Run the command
                print(f'== running singlemodal prediction on source data {self.singlemodal_set}')
                subprocess.run(command_single, env=env)
                print(f'''== saved singlemodal prediction on source data {self.singlemodal_set.parent/(self.singlemodal_set.name+'_predictions')}''')
            else:
                print(F'''== skipping singlemodal prediction, found no files in directory''')

        if '502' in task:
            if os.listdir(self.combined_set):
                command_single = [
                "nnUNet_predict",
                "-i", self.combined_set,
                "-o", self.combined_set.parent/(self.combined_set.name+'_predictions'),
                '-tr', 'nnUNetTrainerV2',
                '-ctr', 'nnUNetTrainerV2CascadeFullRes',
                '-m', '3d_fullres',
                '-p', 'nnUNetPlansv2.1',
                '-t', 'Task502_BrainMetsReseg1to1nodnec'
                ]
                # Run the command
                print(f'== running singlemodal prediction on source data {self.combined_set}')
                subprocess.run(command_single, env=env)
                print(f'''== saved singlemodal prediction on source data {self.combined_set.parent/(self.combined_set.name+'_predictions')}''')
            else:
                print(F'''== skipping prediction, found no files in directory''')


#nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2_Loss_DiceCE_noSmooth -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task504_BrainMetsReseg1to3
#nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2_Loss_DiceCE_noSmooth -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task524_BrainMetsResegMultimod1to3
