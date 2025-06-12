import subprocess
import os
import pathlib as pl
import time

class Resegmentor():
    """
    Resegmentation processor
    """
    def execute(self, task=['524', '504'], nnUNet_dir = "/home/lorenz/BMPipeline/resegmentation"):
        """
        Execute the processor
        task = string or list of strings, the conversion mode. if string only does one mode, if list doess all the modes in the string
        nnUNet_dir = string, the path to the direcotry that stores the nnUNet outputs

        502 = singleclass output no t2
        504 = multiclass output no t2
        524 = multiclass outptu with t2
        """
        env = os.environ.copy()
        env["RESULTS_FOLDER"] = nnUNet_dir ## important: this is the directory that needs to contain 
        time.sleep(1)
        if '524' in task:
            if os.listdir(self.set524):
                command_multi = [
                "nnUNet_predict",
                "-i", self.set524,
                "-o", self.set524.parent/(self.set524.name+'_predictions'),
                '-tr', 'nnUNetTrainerV2_Loss_DiceCE_noSmooth',
                '-ctr', 'nnUNetTrainerV2CascadeFullRes',
                '-m', '3d_fullres',
                '-p', 'nnUNetPlansv2.1',
                '-t', 'Task524_BrainMetsResegMultimod1to3'
                ]
                # Run the command
                print(f'== running multimodal prediction on source data {self.set524}')
                subprocess.run(command_multi, env=env)
                print(f'''== saved multimodal prediction on source data {self.set524.parent/(self.set524.name+'_predictions')}''')
            else:
                print('== skipping multimodal prediction, found no files in directory')


        if '504' in task:
            if os.listdir(self.set504):
                command_single = [
                "nnUNet_predict",
                "-i", self.set504,
                "-o", self.set504.parent/(self.set504.name+'_predictions'),
                '-tr', 'nnUNetTrainerV2_Loss_DiceCE_noSmooth',
                '-ctr', 'nnUNetTrainerV2CascadeFullRes',
                '-m', '3d_fullres',
                '-p', 'nnUNetPlansv2.1',
                '-t', 'Task504_BrainMetsReseg1to3'
                ]
                # Run the command
                print(f'== running singlemodal prediction on source data {self.set504}')
                subprocess.run(command_single, env=env)
                print(f'''== saved singlemodal prediction on source data {self.set504.parent/(self.set504.name+'_predictions')}''')
            else:
                print(F'''== skipping singlemodal prediction, found no files in directory''')

        if '502' in task:
            if os.listdir(self.set502):
                command_single = [
                "nnUNet_predict",
                "-i", self.set502,
                "-o", self.set502.parent/(self.set502.name+'_predictions'),
                '-tr', 'nnUNetTrainerV2',
                '-ctr', 'nnUNetTrainerV2CascadeFullRes',
                '-m', '3d_fullres',
                '-p', 'nnUNetPlansv2.1',
                '-t', 'Task502_BrainMetsReseg1to1nodnec'
                ]
                # Run the command
                print(f'== running singlemodal prediction on source data {self.set502}')
                subprocess.run(command_single, env=env)
                print(f'''== saved singlemodal prediction on source data {self.set502.parent/(self.set502.name+'_predictions')}''')
            else:
                print(F'''== skipping prediction, found no files in directory''')


#nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2_Loss_DiceCE_noSmooth -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task504_BrainMetsReseg1to3
#nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2_Loss_DiceCE_noSmooth -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task524_BrainMetsResegMultimod1to3
