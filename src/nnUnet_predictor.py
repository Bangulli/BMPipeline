import subprocess
import os
import pathlib as pl

class Resegmentor():
    def __init__(self, multimodal_set, singlemodal_set):
        self.multimodal_set = multimodal_set
        self.singlemodal_set = singlemodal_set

    def execute(self):
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
            subprocess.run(command_multi)
            print(f'''== saved multimodal prediction on source data {self.multimodal_set.parent/(self.multimodal_set.name+'_predictions')}''')
        else:
            print('== skipping multimodal prediction, found no files in directory')

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
            subprocess.run(command_single)
            print(f'''== saved singlemodal prediction on source data {self.singlemodal_set.parent/(self.singlemodal_set.name+'_predictions')}''')
        else:
            print(F'''== skipping singlemodal prediction, found no files in directory''')

#nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2_Loss_DiceCE_noSmooth -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task504_BrainMetsReseg1to3
#nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2_Loss_DiceCE_noSmooth -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task524_BrainMetsResegMultimod1to3
